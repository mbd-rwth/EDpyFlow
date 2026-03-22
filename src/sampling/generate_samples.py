"""
Step 1 — Sampling

Generates building configurations using Latin Hypercube Sampling (LHS) for
four TABULA DE building typologies (SFH, TH, MFH, AB), then expands the
samples across all locations and refurbishment levels defined in config.yaml.

Output: runs/{run_name}/samples.csv
        runs/{run_name}/config.yaml
"""

import shutil
import sys
import numpy as np
import pandas as pd
import yaml
from itertools import product
from psimpy.sampler import LHS
import os


# Sampling bounds per building typology, derived from TABULA DE reference buildings.
# Each row defines [min, max] for one input parameter.
# Parameters (columns): construction_year, net_leased_area [m²], num_floors, floor_height [m]
BUILDING_BOUNDS = {
    "MFH": np.array([[1850, 2015], [200, 2300], [2, 6],  [2.5, 4]]),
    "SFH": np.array([[1850, 2015], [50,  280],  [1, 3],  [2.5, 4]]),
    "TH":  np.array([[1860, 2015], [50,  280],  [1, 3],  [2.5, 4]]),
    "AB":  np.array([[1860, 1978], [500, 2900], [5, 10], [2.5, 4]]),
}


def sample_buildings(building_type, bounds, n_samples, seed, criterion):
    """
    Generate LHS samples for a single building typology.

    Returns a DataFrame with columns:
        construction_year, net_leased_area, num_floors, floor_height, building_type
    """
    sampler = LHS(ndim=bounds.shape[0], bounds=bounds, seed=seed, criterion=criterion)
    samples = sampler.sample(nsamples=n_samples)

    # Discrete parameters rounded to nearest integer
    samples[:, 0] = np.round(samples[:, 0])  # construction_year
    samples[:, 2] = np.round(samples[:, 2])  # num_floors

    # Continuous parameters rounded to one decimal place
    samples[:, 1] = np.round(samples[:, 1], decimals=1)  # net_leased_area
    samples[:, 3] = np.round(samples[:, 3], decimals=1)  # floor_height

    df = pd.DataFrame(
        samples,
        columns=["construction_year", "net_leased_area", "num_floors", "floor_height"]
    )
    df["building_type"] = building_type
    df[["construction_year", "num_floors"]] = df[["construction_year", "num_floors"]].astype(int)

    return df


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

run_name             = config["run_name"]
n_per_type           = config["sampling"]["n_per_type"]
seed                 = config["sampling"]["seed"]
criterion            = config["sampling"]["criterion"]
locations            = list(config["locations"].keys())
refurbishment_status = config["refurbishment_status"]

run_dir = os.path.join("runs", run_name)

if os.path.exists(run_dir):
    print(f"Error: run '{run_name}' already exists at {run_dir}. Choose a different run_name.")
    sys.exit(1)

os.makedirs(run_dir)
shutil.copy("config.yaml", os.path.join(run_dir, "config.yaml"))

# Generate LHS samples for each building typology
lhs_samples = pd.concat(
    [sample_buildings(btype, bounds, n_per_type, seed, criterion)
     for btype, bounds in BUILDING_BOUNDS.items()],
    ignore_index=True
)

# Expand samples across all (location, refurbishment_status) combinations.
# Each row in the output represents one unique building configuration to simulate.
samples_df = pd.concat(
    [lhs_samples.assign(location=loc, refurbishment_status=ref)
     for loc, ref in product(locations, refurbishment_status)],
    ignore_index=True
)
samples_df.index.name = "id"
samples_df = samples_df.reset_index()[
    ["id", "construction_year", "net_leased_area", "num_floors",
     "floor_height", "building_type", "location", "refurbishment_status"]
]

samples_df.to_csv(os.path.join(run_dir, "samples.csv"), index=False)
print(f"Saved {len(samples_df)} samples to {run_dir}/samples.csv")
