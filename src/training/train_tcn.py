import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

from tcn_model import TCNSurrogate
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

RUN_NAME = config["run_name"]
RUN_DIR = os.path.join("runs", RUN_NAME)

DATA_DIR = os.path.join(RUN_DIR, "tcn_data")

parser = argparse.ArgumentParser(description="EDpyFlow TCN V1/V2 training")
parser.add_argument("--version", choices=["v1", "v2"], default="v1")
parser.add_argument("--train-seed", type=int, default=42)
parser.add_argument("--split-seed", type=int, default=42)
parser.add_argument("--model-dir", type=str, default=None)
args = parser.parse_args()

if args.model_dir is None:
    raise ValueError(
        "Please provide --model-dir to keep the original model files safe."
    )

MODEL_DIR = os.path.abspath(args.model_dir)

protected_dirs = [
    os.path.abspath(os.path.join(RUN_DIR, "models")),
    os.path.abspath(os.path.join(RUN_DIR, "models_v2")),
]

if MODEL_DIR in protected_dirs:
    raise ValueError(
        "This directory contains an original TCN model. "
        "Please choose a different --model-dir."
    )

TRAIN_SEED = args.train_seed
SPLIT_SEED = args.split_seed

EPOCHS = 300
LEARNING_RATE = 0.001
BATCH_SIZE = 2
PATIENCE = 30

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    X_weather = np.load(
        os.path.join(DATA_DIR, "X_weather.npy")
    ).astype(np.float32)

    X_static = np.load(
        os.path.join(DATA_DIR, "X_static.npy")
    ).astype(np.float32)

    y = np.load(
        os.path.join(DATA_DIR, "y.npy")
    ).astype(np.float32)

    ids = np.load(
        os.path.join(DATA_DIR, "ids.npy")
    )

    group_ids = np.load(
        os.path.join(DATA_DIR, "group_ids.npy")
    )

    return X_weather, X_static, y, ids, group_ids



def split_indices(group_ids):
    group_ids = np.asarray(group_ids)

    unique_groups = np.unique(group_ids)

    if len(unique_groups) < 3:
        raise ValueError(
            "At least 3 unique building configurations are required."
        )

    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(unique_groups)

    test_size = config["surrogate"]["test_size"]
    val_size = config["surrogate"]["val_size"]

    n_test = max(1, round(len(unique_groups) * test_size))
    n_val = max(1, round(len(unique_groups) * val_size))
    n_train = len(unique_groups) - n_val - n_test

    if n_train < 1:
        raise ValueError("Not enough building groups for splitting.")

    train_groups = unique_groups[:n_train]
    val_groups = unique_groups[n_train:n_train + n_val]
    test_groups = unique_groups[n_train + n_val:]

    train_idx = np.flatnonzero(np.isin(group_ids, train_groups))
    val_idx = np.flatnonzero(np.isin(group_ids, val_groups))
    test_idx = np.flatnonzero(np.isin(group_ids, test_groups))

    return train_idx, val_idx, test_idx



def standardize_static(train, val, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)

    std[std < 1e-8] = 1.0

    train_scaled = (train - mean) / std
    val_scaled = (val - mean) / std
    test_scaled = (test - mean) / std

    return (
        train_scaled,
        val_scaled,
        test_scaled,
        mean,
        std,
    )


def standardize_weather(train, val, test):


    mean = train.mean(
        axis=(0, 1),
        keepdims=True
    )

    std = train.std(
        axis=(0, 1),
        keepdims=True
    )

    std[std < 1e-8] = 1.0

    train_scaled = (train - mean) / std
    val_scaled = (val - mean) / std
    test_scaled = (test - mean) / std

    return (
        train_scaled,
        val_scaled,
        test_scaled,
        mean,
        std,
    )



def evaluate(
    model,
    weather,
    static,
    y_true,
    y_mean,
    y_std
):
    model.eval()

    predictions_batches = []

    with torch.no_grad():
        for start in range(0, len(y_true), BATCH_SIZE):
            end = start + BATCH_SIZE

            batch_weather = torch.as_tensor(
                weather[start:end],
                dtype=torch.float32,
                device=DEVICE
            )

            batch_static = torch.as_tensor(
                static[start:end],
                dtype=torch.float32,
                device=DEVICE
            )

            batch_predictions = model(
                batch_weather,
                batch_static
            )

            predictions_batches.append(
                batch_predictions.cpu().numpy()
            )

    predictions_scaled = np.concatenate(
        predictions_batches,
        axis=0
    )
    # Convert normalized predictions back to the target scale.
    predictions = (
        predictions_scaled * y_std + y_mean
    )
    # V2 uses log1p, so convert predictions back to kWh.
    if args.version == "v2":
        predictions = np.expm1(predictions)

    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)

    return predictions, rmse, mae, mape, r2


def main():
    set_seed(TRAIN_SEED)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    X_weather, X_static, y, ids, group_ids = load_data()

    train_idx, val_idx, test_idx = split_indices(
        group_ids
    )

    print(
        f"Samples train/val/test: "
        f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}"
    )
    print(
        f"Building groups train/val/test: "
        f"{len(set(group_ids[train_idx]))}/"
        f"{len(set(group_ids[val_idx]))}/"
        f"{len(set(group_ids[test_idx]))}"
    )

    weather_train = X_weather[train_idx]
    weather_val = X_weather[val_idx]
    weather_test = X_weather[test_idx]

    static_train = X_static[train_idx]
    static_val = X_static[val_idx]
    static_test = X_static[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    (
        weather_train,
        weather_val,
        weather_test,
        weather_mean,
        weather_std,
    ) = standardize_weather(
        weather_train,
        weather_val,
        weather_test
    )

    (
        static_train,
        static_val,
        static_test,
        static_mean,
        static_std,
    ) = standardize_static(
        static_train,
        static_val,
        static_test
    )


    if args.version == "v2":
        y_train_target = np.log1p(y_train)
        y_val_target = np.log1p(y_val)
    else:
        y_train_target = y_train
        y_val_target = y_val

    # Calculate normalization values from the training set.
    y_mean = y_train_target.mean()
    y_std = y_train_target.std()

    if y_std < 1e-8:
        y_std = 1.0

    y_train_scaled = (
        y_train_target - y_mean
    ) / y_std

    y_val_scaled = (
        y_val_target - y_mean
    ) / y_std

    model = TCNSurrogate(
        weather_features=X_weather.shape[2],
        static_features=X_static.shape[1],
        channels=(16, 32, 32),
        kernel_size=3,
        dropout=0.2,
    ).to(DEVICE)
    print(f"Using device: {DEVICE}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_function = nn.MSELoss()

    weather_train_tensor = torch.tensor(
        weather_train,
        dtype=torch.float32
    )

    static_train_tensor = torch.tensor(
        static_train,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train_scaled,
        dtype=torch.float32
    )

    weather_val_tensor = torch.tensor(
        weather_val,
        dtype=torch.float32
    )

    static_val_tensor = torch.tensor(
        static_val,
        dtype=torch.float32
    )

    y_val_tensor = torch.tensor(
        y_val_scaled,
        dtype=torch.float32
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    epochs_without_improvement = 0

    for epoch in range(
        1,
        EPOCHS + 1
    ):
        model.train()

        permutation = torch.randperm(
            len(train_idx)
        )

        epoch_loss = 0.0
        batch_count = 0

        for start in range(
            0,
            len(train_idx),
            BATCH_SIZE
        ):
            batch_indices = permutation[
                start:start + BATCH_SIZE
            ]

            batch_weather = weather_train_tensor[
                batch_indices
            ].to(DEVICE)

            batch_static = static_train_tensor[
                batch_indices
            ].to(DEVICE)

            batch_y = y_train_tensor[
                batch_indices
            ].to(DEVICE)

            optimizer.zero_grad()

            predictions = model(
                batch_weather,
                batch_static
            )

            loss = loss_function(
                predictions,
                batch_y
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        mean_train_loss = (
            epoch_loss / batch_count
        )


        model.eval()

        val_loss_total = 0.0
        val_sample_count = 0

        with torch.no_grad():
            for start in range(
                0,
                len(val_idx),
                BATCH_SIZE
            ):
                end = start + BATCH_SIZE

                batch_weather = weather_val_tensor[
                    start:end
                ].to(DEVICE)

                batch_static = static_val_tensor[
                    start:end
                ].to(DEVICE)

                batch_y = y_val_tensor[
                    start:end
                ].to(DEVICE)

                val_predictions = model(
                    batch_weather,
                    batch_static
                )

                batch_loss = loss_function(
                    val_predictions,
                    batch_y
                )

                batch_count = len(batch_y)
                val_loss_total += batch_loss.item() * batch_count
                val_sample_count += batch_count

        val_loss = val_loss_total / val_sample_count


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            # Save best model during training
            checkpoint_path = os.path.join(
                MODEL_DIR,
                "tcn_best_during_training.pt"
            )

            torch.save(
                {
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "weather_mean": weather_mean,
                    "weather_std": weather_std,
                    "static_mean": static_mean,
                    "static_std": static_std,
                    "target_mean": y_mean,
                    "target_std": y_std,
                    "target_transform": "log1p" if args.version == "v2" else "none",
                    "weather_features": X_weather.shape[2],
                    "static_features": X_static.shape[1],
                    "static_feature_names": [
                        "construction_year",
                        "net_leased_area",
                        "num_floors",
                        "floor_height",
                        "building_type_MFH",
                        "building_type_SFH",
                        "building_type_TH",
                        "building_type_AB",
                        "refurbishment_status_standard",
                        "refurbishment_status_retrofit",
                        "refurbishment_status_adv_retrofit",
                    ],
                    "channels": (16, 32, 32),
                    "kernel_size": 3,
                    "dropout": 0.2,
                    "random_state": SPLIT_SEED,
                    "training_seed": TRAIN_SEED,
                },
                checkpoint_path
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epoch == 1
            or epoch % 25 == 0
        ):
            print(
                f"Epoch {epoch:3d} | "
                f"Train loss: {mean_train_loss:.6f} | "
                f"Val loss: {val_loss:.6f}"
            )

        if (
            epochs_without_improvement
            >= PATIENCE
        ):
            print()
            print(
                f"Early stopping at epoch {epoch}"
            )
            break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    print()
    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print()

    (
        predictions,
        rmse,
        mae,
        mape,
        r2,
    ) = evaluate(
        model,
        weather_test,
        static_test,
        y_test,
        y_mean,
        y_std
    )

    print("Test results")
    print("------------")

    print("First 10 test predictions:")

    for building_id, actual, predicted in zip(
        ids[test_idx][:10],
        y_test[:10],
        predictions[:10]
    ):
        print(
            f"Building {building_id}: "
            f"actual = {actual:.2f} kWh, "
            f"predicted = {predicted:.2f} kWh"
        )

    print()
    print(
        f"RMSE: {rmse:.2f} kWh"
    )
    print(
        f"MAE:  {mae:.2f} kWh"
    )
    print(
        f"MAPE: {mape * 100:.2f}%"
    )
    print(
        f"R²:   {r2:.4f}"
    )

    model_path = os.path.join(
        MODEL_DIR,
        "tcn_surrogate_base.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "weather_mean":
                weather_mean,

            "weather_std":
                weather_std,

            "static_mean":
                static_mean,

            "static_std":
                static_std,

            "target_mean":
                y_mean,

            "target_std":
                y_std,

            "target_transform":
                "log1p" if args.version == "v2" else "none",

            "weather_features":
                X_weather.shape[2],

            "static_features":
                X_static.shape[1],

            "static_feature_names": [
                "construction_year",
                "net_leased_area",
                "num_floors",
                "floor_height",
                "building_type_MFH",
                "building_type_SFH",
                "building_type_TH",
                "building_type_AB",
                "refurbishment_status_standard",
                "refurbishment_status_retrofit",
                "refurbishment_status_adv_retrofit",
            ],

            "channels":
                (16, 32, 32),

            "kernel_size":
                3,

            "dropout":
                0.2,

            "best_epoch":
                best_epoch,

            "random_state":
                SPLIT_SEED,

            "training_seed":
                TRAIN_SEED,
        },
        model_path
    )

    print()
    print(
        f"Saved model to: "
        f"{model_path}"
    )


if __name__ == "__main__":
    main()