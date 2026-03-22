"""
Step 2 — Thermal Model Generation

For each location in samples.csv, generates reduced-order building thermal
models in Modelica using TEASER and exports them as AixLib-compatible packages
for OpenModelica simulation.

Reads:  runs/{run_name}/samples.csv, config.yaml
Output: runs/{run_name}/simulation_input/residentials_{location}/
"""

import os
import pandas as pd
import yaml
from teaser.project import Project


# TABULA DE geometry identifiers per building typology
GEOMETRY_DATA = {
    "MFH": "tabula_de_multi_family_house",
    "SFH": "tabula_de_single_family_house",
    "TH":  "tabula_de_terraced_house",
    "AB":  "tabula_de_apartment_block",
}


def add_buildings_to_project(project, group):
    """Add all buildings in a location group to a TEASER project."""
    for _, row in group.iterrows():
        project.add_residential(
            construction_data=f"tabula_de_{row['refurbishment_status']}",
            geometry_data=GEOMETRY_DATA[row["building_type"]],
            name=f"residential_{row['id']}",
            year_of_construction=row["construction_year"],
            number_of_floors=row["num_floors"],
            height_of_floors=row["floor_height"],
            net_leased_area=row["net_leased_area"],
        )


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

run_name  = config["run_name"]
num_elements = config["num_elements"]
run_dir   = os.path.join("runs", run_name)

samples = pd.read_csv(os.path.join(run_dir, "samples.csv"))

for location, group in samples.groupby("location"):

    # TEASER requires absolute path
    weather_file = os.path.abspath(
        os.path.join("data", "locations", config["locations"][location])
    )

    # TEASER requires absolute path
    output_dir = os.path.abspath(
        os.path.join(run_dir, "simulation_input")
    )
    os.makedirs(output_dir, exist_ok=True)

    prj = Project()
    prj.name = f"residentials_{location}"

    add_buildings_to_project(prj, group)

    prj.used_library_calc = "AixLib"
    prj.number_of_elements_calc = num_elements
    prj.weather_file_path = weather_file

    prj.calc_all_buildings()
    prj.export_aixlib(internal_id=None, path=output_dir)

    print(f"Generated: {location}")
