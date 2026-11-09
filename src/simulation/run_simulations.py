"""
Step 3 — OpenModelica Simulation

Runs building energy simulations for all Modelica models generated in Step 2.
Results are the annual heat demand (kWh) per building together with the runtime
of the simulation that produced it, stored as JSON files keyed by building id.

A simulation counts as successful only if its output reaches the configured
stop time. Incomplete runs are retried up to max_attempts times; buildings that
never complete are written to failed_{location}.csv and left out of the results.

Supports resume: if a partial sim_results_{location}.json already exists,
already-simulated buildings are skipped.

Reads:  config.yaml, runs/{run_name}/samples.csv
        runs/{run_name}/simulation_input/residentials_{location}/
Output: runs/{run_name}/simulation_output/sim_results_{location}.json
        runs/{run_name}/simulation_output/failed_{location}.csv
        runs/{run_name}/logs/simulation_{timestamp}.log
"""

from OMPython import OMCSessionZMQ
import pandas as pd
import numpy as np
import logging
import json
import gzip
import shutil
import re
import os
import yaml
from datetime import datetime
from time import perf_counter


# Path to AixLib within the container
AIXLIB_FILE = "/opt/AixLib/AixLib/package.mo"

# Compression level for retained result files.
COMPRESS_LEVEL = 1

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

run_name            = config["run_name"]
locations           = list(config["locations"].keys())
stop_time           = config["simulation"]["duration_days"] * 86400
intervals           = config["simulation"]["duration_days"] * 24 // config["simulation"]["timestep_hours"]
keep_raw_output     = config["simulation"].get("keep_raw_output", False)
compress_raw_output = config["simulation"].get("compress_raw_output", False)
max_attempts        = config["simulation"].get("max_attempts", 3)

# A run counts as complete when its last output point is within one interval of stopTime.
completion_tolerance = stop_time / intervals

run_dir        = os.path.join("runs", run_name)
sim_input_dir  = os.path.abspath(os.path.join(run_dir, "simulation_input"))
sim_output_dir = os.path.join(run_dir, "simulation_output")
logs_dir = os.path.join(run_dir, "logs")

os.makedirs(sim_output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, f"simulation_{datetime.now():%Y%m%d_%H%M%S}.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def compress_result_file(path):
    """Replace a raw result file with its gzip."""
    with open(path, "rb") as raw, \
         gzip.open(f"{path}.gz", "wb", compresslevel=COMPRESS_LEVEL) as packed:
        shutil.copyfileobj(raw, packed, length=1024 * 1024)
    os.remove(path)


samples = pd.read_csv(os.path.join(run_dir, "samples.csv"))

omc = OMCSessionZMQ()

assert omc.sendExpression("loadModel(Modelica)"), omc.sendExpression("getErrorString()")
omc.sendExpression(f'loadFile("{AIXLIB_FILE}")')

for lib in omc.sendExpression("getClassNames()"):
    logger.info(f"Library: {lib}, Version: {omc.sendExpression(f'getVersion({lib})')}")

for location in locations:

    results_path = os.path.join(sim_output_dir, f"sim_results_{location}.json")
    failed_path  = os.path.join(sim_output_dir, f"failed_{location}.csv")

    # Resume: load existing results if available
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        logger.info(f"{location}: resuming with {len(results)} buildings already done.")
    else:
        results = {}

    # A building already recorded as failed is not attempted again.
    if os.path.exists(failed_path):
        failures = pd.read_csv(failed_path).to_dict("records")
        logger.info(f"{location}: {len(failures)} buildings already recorded as failed.")
    else:
        failures = []

    failed_ids = {int(record["id"]) for record in failures}

    directory = os.path.abspath(
        os.path.join(sim_input_dir, f"residentials_{location}")
    )

    # Temporary directory for raw simulation output files
    temp_dir = os.path.join(directory, "Outputs")
    os.makedirs(temp_dir, exist_ok=True)
    temp_dir = temp_dir.replace(os.sep, "/")

    root_package = os.path.join(directory, "package.mo").replace(os.sep, "/")
    if not omc.sendExpression(f'loadFile("{root_package}")'):
        logger.error(f"Failed to load package: {root_package}")
        exit()

    package_order_file = os.path.join(directory, "package.order")
    if not os.path.exists(package_order_file):
        logger.error(f"package.order not found in {directory}")
        exit()

    with open(package_order_file, "r") as f:
        sub_packages = [line.strip() for line in f if line.strip()]

    omc.sendExpression(f'cd("{temp_dir}")')

    n_success = 0
    n_failed  = 0

    for sub_package in sub_packages:

        # Extract the building id from the package name (e.g. "residential_42" -> "42")
        id_match = re.search(r"residential_(\d+)", sub_package)
        if not id_match:
            logger.warning(f"Could not extract id from: {sub_package}")
            continue

        building_id = id_match.group(1)

        if building_id in results or int(building_id) in failed_ids:
            continue

        model_name  = f"residentials_{location}.{sub_package}.{sub_package}"
        result_file = os.path.join(temp_dir, model_name) + "_res.csv"

        # reason holds why the last attempt failed, and is None once one succeeds.
        reason = "not attempted"

        for attempt in range(1, max_attempts + 1):

            started = perf_counter()
            sim_result = omc.sendExpression(
                f'simulate({model_name}, stopTime={stop_time}, '
                f'numberOfIntervals={intervals}, outputFormat="csv")'
            )
            wall_time = perf_counter() - started

            if not sim_result:
                reason = f"no result from omc: {omc.sendExpression('getErrorString()')}"
            elif sim_result.get("messages", "").startswith("Failed to build model"):
                reason = f"failed to build model: {omc.sendExpression('getErrorString()')}"
            elif not os.path.exists(result_file):
                reason = "simulation produced no output file"
            else:
                df      = pd.read_csv(result_file)
                seconds = df["time"].values
                if seconds[-1] < stop_time - completion_tolerance:
                    reason = (f"stopped at day {seconds[-1] / 86400:.1f} "
                              f"of {stop_time / 86400:.0f}")
                else:
                    reason = None
                    break

            logger.warning(f"{sub_package}: attempt {attempt} of {max_attempts} — {reason}")

        if reason is None:
            # Integrate heating power over time to get annual heat demand [kWh]
            power = df["multizone.zone[1].PHeater"].values
            total_energy = np.trapezoid(power, seconds) / 3_600_000  # W·s (J) → kWh

            # Timing keys are those returned by OpenModelica's simulate(); a key the
            # installed version does not return is recorded as null.
            results[building_id] = {
                "total_energy": total_energy,
                "days_simulated": seconds[-1] / 86400,
                "attempts": attempt,
                "wall_time_s": round(wall_time, 3),
                "compile_time_s": sim_result.get("timeCompile"),
                "simulation_time_s": sim_result.get("timeSimulation"),
                "total_time_s": sim_result.get("timeTotal"),
            }
            n_success += 1
            logger.info(f"{sub_package}: {total_energy:.2f} kWh in {wall_time:.1f} s")

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
        else:
            logger.error(f"{sub_package}: failed after {max_attempts} attempts — {reason}")
            sample = samples.loc[samples["id"] == int(building_id)].iloc[0].to_dict()
            failures.append({**sample, "attempts": max_attempts, "reason": reason})
            n_failed += 1

            pd.DataFrame(failures).to_csv(failed_path, index=False)

        if keep_raw_output:
            if compress_raw_output and os.path.exists(result_file):
                compress_result_file(result_file)
        else:
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    logger.info(f"{location}: done — {n_success} succeeded, {n_failed} failed.")
