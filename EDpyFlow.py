#!/usr/bin/env python3
"""EDpyFlow Orchestrator"""

import os
import shutil
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime

# All paths resolve relative to this script — works locally and inside container
ROOT = Path(__file__).parent
logger = logging.getLogger(__name__)

ENV = "/opt/micromamba/envs"

STAGES = [
    ("sampling",           f"{ENV}/sampling/bin/python",  "src/sampling/generate_samples.py",        "LHS Parameter Sampling"),
    ("modeling",           f"{ENV}/modeling/bin/python",  "src/modeling/generate_thermal_models.py",  "TEASER Model Generation"),
    ("simulation",         f"{ENV}/simulate/bin/python",  "src/simulation/run_simulations.py",        "OpenModelica Simulation"),
    ("dataset_assembly",   f"{ENV}/surrogate/bin/python", "src/data_prep/generate_dataset.py",        "Dataset Preparation"),
    ("surrogate_training", f"{ENV}/surrogate/bin/python", "src/training/train_surrogate.py",          "Surrogate Model Training"),
]

STAGE_MAP = {name: (python, script, desc) for name, python, script, desc in STAGES}


# ---------------------------------------------------------------------------
# Container boundary — transparent to the user
# ---------------------------------------------------------------------------

def _inside_container() -> bool:
    """Detect whether we are already running inside an Apptainer/Singularity container."""
    return os.path.exists("/.singularity.d")


def _relaunch_in_container() -> None:
    """Re-exec the current invocation transparently through Apptainer."""
    sif = ROOT / "container" / "EDpyFlow.sif"

    if not shutil.which("apptainer"):
        sys.exit(
            "Error: Apptainer is not installed or not on PATH.\n"
            "See: https://apptainer.org/docs/user/latest/quick_start.html"
        )

    if not sif.exists():
        sys.exit(
            f"Error: Container image not found at {sif}\n"
            "Build it first with:\n"
            "  cd container && apptainer build EDpyFlow.sif EDpyFlow.def"
        )

    apptainer = shutil.which("apptainer")
    os.execv(apptainer, [
        "apptainer", "run",
        "--bind", f"{ROOT}:/app",
        "--pwd", "/app",
        str(sif),
        *sys.argv[1:],          # forward all arguments unchanged
    ])
    # os.execv replaces the current process — nothing below this runs


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class WorkflowOrchestrator:
    def __init__(self):
        self.results = []
        self._ensure_dirs()
        ui.print_banner(ROOT, ROOT / "data")

    def _ensure_dirs(self):
        for path in (
            ROOT / "data" / "locations",
            ROOT / "runs",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def run_stage(self, name: str, python: str, script: str, description: str) -> bool:
        script_path = ROOT / script
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            self._record(description, success=False, duration=None)
            return False

        cmd = [python, str(script_path)]
        ui.print_rule(description)
        ui.print_cmd(cmd)

        start = datetime.now()
        success = False

        with ui.Spinner(description):
            try:
                subprocess.run(cmd, check=True, cwd=ROOT)
                success = True
            except subprocess.CalledProcessError as e:
                logger.error(f"Stage failed with exit code {e.returncode}")

        duration = datetime.now() - start
        self._record(description, success=success, duration=duration)
        ui.print_stage_result(success, duration)
        return success

    def _record(self, description, success, duration):
        self.results.append({
            "description": description,
            "success": success,
            "duration": str(duration).split(".")[0] if duration else "—",
        })

    def run_full_workflow(self) -> bool:
        logger.info("Starting full workflow")
        start = datetime.now()

        for name, python, script, description in STAGES:
            if not self.run_stage(name, python, script, description):
                ui.print_summary(self.results)
                logger.error("Workflow stopped due to error")
                return False

        ui.print_summary(self.results)
        ui.print_workflow_success(datetime.now() - start)
        return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Transparently relaunch through Apptainer if running outside the container.
    # Users simply call `python EDpyFlow.py` — no wrapper script needed.
    if not _inside_container():
        _relaunch_in_container()

    # ui imports rich — only safe after the container check above,
    # since rich is installed in the container but not on the host.
    global ui
    import ui
    import yaml

    with open("config.yaml") as f:
        run_name = yaml.safe_load(f)["run_name"]

    import argparse

    parser = argparse.ArgumentParser(description="Run EDpyFlow pipeline")
    parser.add_argument("--stage", choices=list(STAGE_MAP.keys()),
                        help=f"Run a single stage: {', '.join(STAGE_MAP.keys())}")
    args = parser.parse_args()

    run_dir = os.path.join("runs", run_name)
    # A single stage runs inside an existing run, so only guard against
    # overwriting when starting a fresh full run.
    if args.stage is None and os.path.exists(run_dir):
        sys.exit(f"Error: run '{run_name}' already exists at {run_dir}. Choose a different run_name in config.yaml.")

    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            ui.get_log_handler(),
            logging.FileHandler(os.path.join(logs_dir, f"workflow_{datetime.now():%Y%m%d_%H%M%S}.log")),
        ],
    )

    try:
        orchestrator = WorkflowOrchestrator()

        if args.stage:
            python, script, desc = STAGE_MAP[args.stage]
            orchestrator.run_stage(args.stage, python, script, desc)
            ui.print_summary(orchestrator.results)
            success = orchestrator.results[-1]["success"]
        else:
            success = orchestrator.run_full_workflow()

        sys.exit(0 if success else 1)

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()