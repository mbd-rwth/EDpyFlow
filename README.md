# EDpyFlow

![Python](https://img.shields.io/badge/python-3.10-blue)
![Apptainer](https://img.shields.io/badge/container-Apptainer-informational)

**EDpyFlow** is a containerized, end-to-end Python workflow for generating building energy simulations and training surrogate models for residential heat-demand prediction. It builds on [TEASER](https://github.com/RWTH-EBC/TEASER) and [OpenModelica](https://openmodelica.org/) and bundles all required dependencies in an Apptainer container, including OpenModelica, [AixLib](https://github.com/RWTH-EBC/AixLib), TEASER, and the necessary Python environments. This allows users to run the full workflow without prior experience with Modelica tooling.

## Pipeline

The pipeline proceeds in five stages. Each stage is self-contained and uses file-based data exchange. The pipeline can be entered or interrupted at any point without reprocessing upstream results.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `src/sampling/generate_samples.py` | LHS sampling of building configurations |
| 2 | `src/modeling/generate_thermal_models.py` | Generates thermal models in Modelica using TEASER |
| 3 | `src/simulation/run_simulations.py` | Runs annual energy simulations in OpenModelica |
| 4 | `src/data_prep/generate_dataset.py` | Assembles simulation results into a dataset |
| 5 | `src/training/train_surrogate.py` | Trains an XGBoost surrogate model |

## Requirements

- [Apptainer](https://apptainer.org/) to build and run the container
- Weather files in `.mos` format for the six locations (see [`data/locations/README.md`](data/locations/README.md))

Build the container before first use:

```bash
cd container && apptainer build EDpyFlow.sif EDpyFlow.def
```

## Configuration

All parameters are set in `config.yaml`:

- `run_name` — name of the run; outputs are written to `runs/{run_name}/`
- `locations` — city names and their weather files
- `refurbishment_status` — refurbishment levels to simulate
- `sampling` — LHS parameters (samples per typology, seed, criterion)
- `num_elements` — number of RC elements in the thermal model
- `simulation` — simulation duration, timestep, and optional raw output retention
- `surrogate` — XGBoost hyperparameters, train/val/test split ratios, and model name

> **Note:** Change `run_name` for each new run to avoid overwriting previous results.

## Usage

Run the full pipeline:

```bash
python EDpyFlow.py
```


## Output

All outputs are written to `runs/{run_name}/`:

```
runs/{run_name}/
├── config.yaml                         ← copy of config at time of run
├── samples.csv                         ← building configurations (Step 1)
├── simulation_input/                   ← Modelica packages (Step 2)
│   ├── residentials_berlin/
│   └── ...
├── simulation_output/                  ← simulation results (Step 3)
│   ├── sim_results_berlin.json
│   └── ...
├── logs/
│   ├── simulation_{timestamp}.log      ← simulation log (Step 3)
│   └── workflow_{timestamp}.log        ← workflow log
├── synthetic_dataset/
│   └── dataset.csv                     ← training dataset (Step 4)
└── models/
    └── {model_name}.json               ← trained surrogate model (Step 5)
```

## Contact

For questions, please contact [bagherinejad@mbd.rwth-aachen.de](mailto:bagherinejad@mbd.rwth-aachen.de) or open an issue at [https://github.com/mbd-rwth/EDpyFlow/issues](https://github.com/mbd-rwth/EDpyFlow/issues).

## Contributors

This project was developed by Nazanin Bagherinejad, with contributions from:

- [V Mithlesh Kumar] - Apptainer containerization

## License

This repository is licensed under the [MIT License](LICENSE).

This repository includes an Apptainer definition file used to build the container environment. Third-party software installed during the build remains subject to its respective licenses. Users are responsible for ensuring compliance with those licenses when building or redistributing container images.

## Acknowledgments

This work was performed as part of the ENERsyte project and received funding from Innovationsförderagentur.NRW through the Grüne Gründungen.NRW initiative of the Ministry for the Environment, Nature Conservation and Transport of the State of North Rhine-Westphalia within the framework of the EFRE/JTF-Programme NRW 2021-2027, Co-funded by the European Union (EFRE-20800324).