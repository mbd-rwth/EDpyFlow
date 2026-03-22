"""
Step 4 — Dataset Assembly

Combines the building configurations from samples.csv with the simulation
results from Step 3 to produce the final synthetic dataset used for
surrogate model training.

Reads:  runs/{run_name}/samples.csv, config.yaml
        runs/{run_name}/simulation_output/sim_results_{location}.json
Output: runs/{run_name}/synthetic_dataset/dataset.csv
"""

import os
import json
import pandas as pd
import yaml


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

run_name  = config["run_name"]
locations = list(config["locations"].keys())

run_dir        = os.path.join("runs", run_name)
sim_output_dir = os.path.join(run_dir, "simulation_output")
output_dir     = os.path.join(run_dir, "synthetic_dataset")

samples = pd.read_csv(os.path.join(run_dir, "samples.csv"))
samples["total_energy"] = None

for location in locations:

    res_file = os.path.join(sim_output_dir, f"sim_results_{location}.json")

    with open(res_file, "r") as f:
        results = json.load(f)

    for building_id, energy in results.items():
        samples.loc[samples["id"] == int(building_id), "total_energy"] = energy

os.makedirs(output_dir, exist_ok=True)
samples.to_csv(os.path.join(output_dir, "dataset.csv"), index=False)

print(f"Saved {len(samples)} rows to {output_dir}/dataset.csv")
