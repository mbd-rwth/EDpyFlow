"""
Step 3 — OpenModelica Simulation

Runs building energy simulations for all Modelica models generated in Step 2.
Results are the annual heat demand (kWh) per building, stored as JSON files
keyed by building id.

Supports resume: if a partial sim_results_{location}.json already exists,
already-simulated buildings are skipped.

Reads:  config.yaml
        runs/{run_name}/simulation_input/residentials_{location}/
Output: runs/{run_name}/simulation_output/sim_results_{location}.json
        runs/{run_name}/simulation.log
"""

from OMPython import OMCSessionZMQ
import pandas as pd
import numpy as np
import logging
import json
import re
import os
import yaml


# Path to AixLib within the container
AIXLIB_FILE = "/opt/AixLib/AixLib/package.mo"

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

run_name        = config["run_name"]
locations       = list(config["locations"].keys())
stop_time       = config["simulation"]["duration_days"] * 86400
intervals       = config["simulation"]["duration_days"] * 24 // config["simulation"]["timestep_hours"]
keep_raw_output = config["simulation"].get("keep_raw_output", False)

run_dir        = os.path.join("runs", run_name)
sim_input_dir  = os.path.abspath(os.path.join(run_dir, "simulation_input"))
sim_output_dir = os.path.join(run_dir, "simulation_output")

os.makedirs(sim_output_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(run_dir, "simulation.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

omc = OMCSessionZMQ()

assert omc.sendExpression("loadModel(Modelica)"), omc.sendExpression("getErrorString()")
omc.sendExpression(f'loadFile("{AIXLIB_FILE}")')

for lib in omc.sendExpression("getClassNames()"):
    logger.info(f"Library: {lib}, Version: {omc.sendExpression(f'getVersion({lib})')}")

for location in locations:

    results_path = os.path.join(sim_output_dir, f"sim_results_{location}.json")

    # Resume: load existing results if available
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        logger.info(f"{location}: resuming with {len(results)} buildings already done.")
    else:
        results = {}

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

        if building_id in results:
            continue

        model_name = f"residentials_{location}.{sub_package}.{sub_package}"

        sim_result = omc.sendExpression(
            f'simulate({model_name}, stopTime={stop_time}, '
            f'numberOfIntervals={intervals}, outputFormat="csv")'
        )

        if not sim_result or sim_result.get("messages", "").startswith("Failed to build model"):
            logger.error(f"Simulation failed for {sub_package}: {omc.sendExpression('getErrorString()')}")
            n_failed += 1
            continue

        # Integrate heating power over time to get annual heat demand [kWh]
        df    = pd.read_csv(os.path.join(temp_dir, model_name) + "_res.csv")
        time  = df["time"].values
        power = df["multizone.zone[1].PHeater"].values
        total_energy = np.trapezoid(power, time) / 3_600_000  # W·s (J) → kWh

        results[building_id] = total_energy
        n_success += 1
        logger.info(f"{sub_package}: {total_energy:.2f} kWh")

        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        if not keep_raw_output:
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    logger.info(f"{location}: done — {n_success} succeeded, {n_failed} failed.")
