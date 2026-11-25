import os
import json
import numpy as np
import pandas as pd
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

RUN_NAME = config["run_name"]

RUN_DIR = os.path.join("runs", RUN_NAME)

DATASET_PATH = os.path.join(
    RUN_DIR,
    "synthetic_dataset",
    "dataset_valid.csv"
)

WEATHER_DIR = os.path.join("data", "locations")

OUTPUT_DIR = os.path.join(
    RUN_DIR,
    "tcn_data"
)

EXPECTED_WEATHER_STEPS = 8760


# Zero-based indexes in the 30-column MOS weather table.
# Some columns are only needed to calculate TBlaSky.
MOS_COLUMNS = {
    "dry_bulb_temperature": 1,
    "dew_point_temperature": 2,
    "relative_humidity": 3,
    "global_horizontal_radiation": 8,
    "direct_normal_radiation": 9,
    "diffuse_horizontal_radiation": 10,
    "opaque_sky_cover": 18,
}


# Final temporal features passed to the TCN.
WEATHER_FEATURES = [
    "dry_bulb_temperature",
    "relative_humidity",
    "black_body_sky_temperature",
    "global_horizontal_radiation",
    "diffuse_horizontal_radiation",
    "direct_normal_radiation",
]


WEATHER_UNITS = {
    "dry_bulb_temperature": "K",
    "relative_humidity": "1",
    "black_body_sky_temperature": "K",
    "global_horizontal_radiation": "W/m2",
    "diffuse_horizontal_radiation": "W/m2",
    "direct_normal_radiation": "W/m2",
}


def read_mos_file(file_path):
    rows = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if line.lower().startswith("double"):
                continue

            values = line.replace(",", " ").split()

            try:
                row = [float(value) for value in values]
            except ValueError:
                continue

            if len(row) == 30:
                rows.append(row)

    data = np.asarray(
        rows,
        dtype=np.float32
    )

    if data.ndim != 2:
        raise ValueError(
            f"Could not read weather data from {file_path}"
        )

    if data.shape[1] != 30:
        raise ValueError(
            f"Expected 30 columns in {file_path}, "
            f"found {data.shape[1]}"
        )

    if data.shape[0] != EXPECTED_WEATHER_STEPS:
        raise ValueError(
            f"Expected {EXPECTED_WEATHER_STEPS} hourly rows "
            f"in {file_path}, found {data.shape[0]}"
        )

    return data


def reg_step(y1, y2, x, x_small):
    """
    Smooth approximation of:
        y1 if x > 0
        y2 otherwise

    This follows the regularized step used by the
    Modelica Buildings/AixLib math utilities.
    """
    z = x / x_small

    transition = (
        z
        * (z ** 2 - 3.0)
        * (y2 - y1)
        / 4.0
        + (y1 + y2) / 2.0
    )

    return np.where(
        x > x_small,
        y1,
        np.where(
            x < -x_small,
            y2,
            transition
        )
    )


def smooth_min(x1, x2, delta_x=0.1):
    """
    Smooth approximation of min(x1, x2).
    """
    return reg_step(
        y1=x1,
        y2=x2,
        x=x2 - x1,
        x_small=delta_x
    )


def calculate_black_body_sky_temperature(
    dry_bulb_c,
    dew_point_c,
    opaque_sky_cover_raw
):
    """
    Calculate TBlaSky using the same temperature-and-sky-cover
    formulation used by the Modelica BlackBody model.
    """

    dry_bulb_k = dry_bulb_c + 273.15
    dew_point_k = dew_point_c + 273.15

    # MOS sky cover is stored on a 0...10 scale.
    # ReaderTMY3 converts it to 0...1.
    opaque_sky_cover = np.clip(
        opaque_sky_cover_raw * 0.1,
        0.0,
        1.0
    )

    # Modelica BlackBody uses:
    # smoothMin(TDryBul, TDewPoi, 0.1)
    effective_dew_point_k = smooth_min(
        dry_bulb_k,
        dew_point_k,
        delta_x=0.1
    )

    opaque_sky_cover_10 = (
        10.0 * opaque_sky_cover
    )

    sky_emissivity = (
        0.787
        + 0.764
        * np.log(
            effective_dew_point_k / 273.15
        )
    ) * (
        1.0
        + 0.0224 * opaque_sky_cover_10
        - 0.0035 * opaque_sky_cover_10 ** 2
        + 0.00028 * opaque_sky_cover_10 ** 3
    )

    if np.any(
        ~np.isfinite(sky_emissivity)
    ):
        raise ValueError(
            "Non-finite sky emissivity encountered."
        )

    if np.any(sky_emissivity <= 0):
        raise ValueError(
            "Sky emissivity must be positive."
        )

    black_body_sky_temperature = (
        dry_bulb_k
        * np.power(
            sky_emissivity,
            0.25
        )
    )

    return black_body_sky_temperature.astype(
        np.float32
    )


def load_weather(location):
    file_path = os.path.join(
        WEATHER_DIR,
        f"{location}.mos"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Weather file not found: {file_path}"
        )

    weather_table = read_mos_file(
        file_path
    )

    dry_bulb_c = weather_table[
        :,
        MOS_COLUMNS["dry_bulb_temperature"]
    ]

    dew_point_c = weather_table[
        :,
        MOS_COLUMNS["dew_point_temperature"]
    ]

    relative_humidity = weather_table[
        :,
        MOS_COLUMNS["relative_humidity"]
    ] * 0.01

    relative_humidity = np.clip(
        relative_humidity,
        0.0,
        1.0
    )

    opaque_sky_cover = weather_table[
        :,
        MOS_COLUMNS["opaque_sky_cover"]
    ]

    black_body_sky_temperature = (
        calculate_black_body_sky_temperature(
            dry_bulb_c,
            dew_point_c,
            opaque_sky_cover
        )
    )

    dry_bulb_k = (
        dry_bulb_c + 273.15
    )

    global_horizontal_radiation = weather_table[
        :,
        MOS_COLUMNS[
            "global_horizontal_radiation"
        ]
    ]

    diffuse_horizontal_radiation = weather_table[
        :,
        MOS_COLUMNS[
            "diffuse_horizontal_radiation"
        ]
    ]

    direct_normal_radiation = weather_table[
        :,
        MOS_COLUMNS[
            "direct_normal_radiation"
        ]
    ]

    weather = np.column_stack(
        [
            dry_bulb_k,
            relative_humidity,
            black_body_sky_temperature,
            global_horizontal_radiation,
            diffuse_horizontal_radiation,
            direct_normal_radiation,
        ]
    ).astype(np.float32)

    return weather



BUILDING_TYPES = ["MFH", "SFH", "TH", "AB"]

REFURBISHMENT_LEVELS = [
    "standard",
    "retrofit",
    "adv_retrofit"
]


def prepare_static_features(df):
    numeric_columns = [
        "construction_year",
        "net_leased_area",
        "num_floors",
        "floor_height"
    ]

    categorical_columns = [
        "building_type",
        "refurbishment_status"
    ]

    required_columns = numeric_columns + categorical_columns

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    static_df = df[required_columns].copy()

    # Check numeric features.
    for col in numeric_columns:
        static_df[col] = pd.to_numeric(
            static_df[col],
            errors="raise"
        )

        if not np.isfinite(static_df[col].to_numpy()).all():
            raise ValueError(
                f"Invalid numeric values in: {col}"
            )

    # Check categorical features.
    allowed_categories = {
        "building_type": BUILDING_TYPES,
        "refurbishment_status": REFURBISHMENT_LEVELS
    }

    for col, allowed in allowed_categories.items():
        if static_df[col].isna().any():
            raise ValueError(
                f"Missing values in: {col}"
            )

        unknown = set(static_df[col].unique()) - set(allowed)

        if unknown:
            raise ValueError(
                f"Unknown categories in {col}: {sorted(unknown)}"
            )

    # Convert categorical variables to one-hot features.
    static_df = pd.get_dummies(
        static_df,
        columns=categorical_columns,
        dtype=float
    )

    # Keep the same columns and order for every dataset.
    expected_columns = (
        numeric_columns
        + [f"building_type_{b}" for b in BUILDING_TYPES]
        + [
            f"refurbishment_status_{r}"
            for r in REFURBISHMENT_LEVELS
        ]
    )

    static_df = static_df.reindex(
        columns=expected_columns,
        fill_value=0.0
    )

    return (
        static_df.to_numpy(dtype=np.float32),
        expected_columns
    )



def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df = pd.read_csv(
        DATASET_PATH
    )

    X_static, static_features = (
        prepare_static_features(df)
    )

    weather_sequences = []
    weather_cache = {}

    for _, row in df.iterrows():
        location = row["location"]

        if location not in weather_cache:
            weather_cache[location] = (
                load_weather(location)
            )

            print(
                f"{location}: "
                f"{weather_cache[location].shape}"
            )


        weather_sequences.append(
            weather_cache[location]
        )

    X_weather = np.stack(
        weather_sequences
    ).astype(np.float32)

    y = df[
        "total_energy"
    ].to_numpy(
        dtype=np.float32
    )

    ids = df[
        "id"
    ].to_numpy(
        dtype=np.int64
    )


    # Identify the same building configuration across locations
    # and refurbishment levels.
    group_columns = [
        "construction_year",
        "net_leased_area",
        "num_floors",
        "floor_height",
        "building_type",
    ]

    group_ids = pd.factorize(
        pd.MultiIndex.from_frame(df[group_columns])
    )[0].astype(np.int64)

    print("Unique building groups:", len(np.unique(group_ids)))


    np.save(
        os.path.join(
            OUTPUT_DIR,
            "X_weather.npy"
        ),
        X_weather
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "X_static.npy"
        ),
        X_static
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "y.npy"
        ),
        y
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "ids.npy"
        ),
        ids
    )


    np.save(
        os.path.join(
            OUTPUT_DIR,
            "group_ids.npy"
        ),
        group_ids
    )


    metadata = {
        "num_samples": len(df),
        "weather_features": WEATHER_FEATURES,
        "weather_units": WEATHER_UNITS,
        "static_features": static_features,
        "target": "total_energy",
        "target_unit": "kWh",
        "locations": sorted(
            df["location"].unique().tolist()
        ),
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "metadata.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print()

    print(
        f"X_weather shape: {X_weather.shape}"
    )

    print(
        f"X_static shape:  {X_static.shape}"
    )

    print(
        f"y shape:         {y.shape}"
    )

    print(
        f"ids shape:       {ids.shape}"
    )

    print()

    print(
        f"Weather features: {WEATHER_FEATURES}"
    )

    print(
        f"Static features: {static_features}"
    )

    print(
        f"Target range: "
        f"{y.min():.2f} - "
        f"{y.max():.2f} kWh"
    )

    print()

    print(
        f"Saved TCN data to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()