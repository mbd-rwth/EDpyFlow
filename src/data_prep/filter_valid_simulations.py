"""Filter invalid annual heating-demand records from EDpyFlow datasets."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    data_dir = Path("runs") / args.run_name / "synthetic_dataset"
    source = data_dir / "dataset.csv"
    destination = data_dir / "dataset_valid.csv"

    df = pd.read_csv(source)

    energy = pd.to_numeric(df["total_energy"], errors="coerce")
    valid_mask = np.isfinite(energy) & (energy > 0)

    valid = df.loc[valid_mask].copy()
    valid["total_energy"] = energy.loc[valid_mask]

    print(f"Original simulations: {len(df)}")
    print(f"Valid simulations: {len(valid)}")
    print(f"Excluded simulations: {(~valid_mask).sum()}")

    # Avoid overwriting the existing validated dataset automatically.
    if destination.exists():
        existing = pd.read_csv(destination)
        pd.testing.assert_frame_equal(
            valid.sort_values("id").reset_index(drop=True),
            existing.sort_values("id").reset_index(drop=True),
            check_dtype=False,
        )
        print("PASS: Existing dataset_valid.csv matches the filtering rule.")
    else:
        valid.to_csv(destination, index=False)
        print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
