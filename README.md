# EDpyFlow

![Python](https://img.shields.io/badge/python-3.10-blue)
![Apptainer](https://img.shields.io/badge/container-Apptainer-informational)

**EDpyFlow** is a containerized, end-to-end Python workflow built on TEASER and OpenModelica for running building energy simulations and training surrogate models for predicting residential building heat demand. All dependencies are bundled in the container — OpenModelica, AixLib, TEASER, and the required Python environments — so no prior knowledge of Modelica tooling is required.

In its default configuration, the pipeline covers four TABULA DE building typologies (SFH, TH, MFH, AB), six German cities, and three refurbishment levels, yielding 21,600 building configurations per fidelity level. The resulting trained models constitute the EDSurrogate model family (EDSurrogate-2el and EDSurrogate-4el).

EDpyFlow also ships an interactive **scenario analysis dashboard** for municipal heat planning, enabling planners to explore refurbishment upgrade strategies across a building stock and compare scenarios side by side — powered by the trained surrogate model.

## Pipeline

The pipeline proceeds in five stages. Each stage is self-contained and uses file-based data exchange. The pipeline can be entered or interrupted at any point without reprocessing upstream results.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `src/sampling/generate_samples.py` | LHS sampling of building configurations |
| 2 | `src/modeling/generate_thermal_models.py` | Generates thermal models in Modelica using TEASER |
| 3 | `src/simulation/run_simulations.py` | Runs annual energy simulations in OpenModelica |
| 4 | `src/data_prep/generate_dataset.py` | Assembles simulation results into a dataset |
| 5 | `src/training/train_surrogate.py` | Trains an XGBoost surrogate model |

## Scenario Analysis Dashboard

EDpyFlow includes an interactive scenario analysis tool for municipal heat planning, built on top of the trained surrogate model. It allows users to explore and compare refurbishment upgrade strategies across a building stock — filtering by city, building type, and floor area, and ranking buildings by absolute savings or savings per m².

**Try it:** [EDpyFlow on Hugging Face Spaces](https://huggingface.co/spaces/nazbn/EDpyFlow)

Two analysis modes are available:

- **Single Scenario** — apply an upgrade path (e.g. standard → advanced retrofit) to a filtered stock and visualise energy savings by city, building type, and construction decade, along with a cumulative savings curve
- **Scenario Comparison** — run two scenarios side by side (A vs B) with independent filters and upgrade paths

The dashboard loads any trained model from a run directory and uses the held-out test set saved by `train_surrogate.py` as the default building stock. A custom CSV can also be uploaded.

To run locally:

```bash
streamlit run dashboard/app.py
```

## Requirements

- [Apptainer](https://apptainer.org/) to build and run the container
- Weather files in `.mos` format for the six locations (see [`data/locations/README.md`](data/locations/README.md))

Build the container before first use (from the repo root):

```bash
cd container && apptainer build ../EDpyFlow.sif EDpyFlow.def
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

Run a single step:

```bash
python EDpyFlow.py --step sampling
python EDpyFlow.py --step modeling
python EDpyFlow.py --step simulate
python EDpyFlow.py --step dataset
python EDpyFlow.py --step surrogate
```

A custom container path can be specified with `--container`:

```bash
python EDpyFlow.py --container /path/to/EDpyFlow.sif
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
├── simulation.log                      ← simulation log (Step 3)
├── synthetic_dataset/
│   └── dataset.csv                     ← training dataset (Step 4)
└── models/
    └── {model_name}.json               ← trained surrogate model (Step 5)
```

## Contact

For questions, please contact [bagherinejad@mbd.rwth-aachen.de](mailto:bagherinejad@mbd.rwth-aachen.de) or open an issue at [https://github.com/mbd-rwth/EDpyFlow/issues](https://github.com/mbd-rwth/EDpyFlow/issues).

## Acknowledgments

This work was performed as part of the ENERsyte project and received funding from Innovationsförderagentur.NRW through the Grüne Gründungen.NRW initiative of the Ministry for the Environment, Nature Conservation and Transport of the State of North Rhine-Westphalia within the framework of the EFRE/JTF-Programme NRW 2021-2027, Co-funded by the European Union (EFRE-20800324).