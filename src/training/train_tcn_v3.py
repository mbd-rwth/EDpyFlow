import argparse
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tcn_model import TCNSurrogate


STATIC_FEATURE_NAMES = [
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
]

CHANNELS = (16, 32, 32)
KERNEL_SIZE = 3
DROPOUT = 0.2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="EDpyFlow TCN v3 hybrid-loss training")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lambda-kwh", type=float, default=0.05,
                        help="Weight of standardized kWh-scale MSE in the hybrid loss")
    parser.add_argument("--model-dir", default=None,
                        help="Override output folder (e.g. for smoke testing)")
    parser.add_argument("--train-seed", type=int, default=42,
                    help="Training initialization seed")
    parser.add_argument("--split-seed", type=int, default=42,
                    help="Building-group split seed")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size and patience must be positive")
    if args.lambda_kwh < 0 or not np.isfinite(args.lambda_kwh):
        parser.error("lambda-kwh must be finite and nonnegative")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(group_ids, test_size, val_size, random_state=SEED):
    groups = np.unique(group_ids)
    rng = np.random.default_rng(random_state)
    rng.shuffle(groups)
    n_test = max(1, round(len(groups) * test_size))
    n_val = max(1, round(len(groups) * val_size))
    n_train = len(groups) - n_test - n_val
    if n_train < 1:
        raise ValueError("Not enough building groups for train/validation/test")
    train_groups = groups[:n_train]
    val_groups = groups[n_train:n_train + n_val]
    test_groups = groups[n_train + n_val:]
    train_idx = np.flatnonzero(np.isin(group_ids, train_groups))
    val_idx = np.flatnonzero(np.isin(group_ids, val_groups))
    test_idx = np.flatnonzero(np.isin(group_ids, test_groups))
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(group_ids)
    return train_idx, val_idx, test_idx


def standardize(train, val, test, axes):
    mean = train.mean(axis=axes, keepdims=True)
    std = train.std(axis=axes, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((train - mean) / std, (val - mean) / std,
            (test - mean) / std, mean, std)


def hybrid_loss(pred_scaled, target_scaled, log_mean, log_std,
                energy_mean, energy_std, lambda_kwh):
    # Standardized log1p(kWh) MSE: exactly the v2 loss when lambda_kwh=0.
    log_mse = nn.functional.mse_loss(pred_scaled, target_scaled)
    if lambda_kwh == 0:
        return log_mse, log_mse.detach(), log_mse.new_zeros(())

    # Convert both tensors to physical kWh within the differentiable loss.
    # A bounded exponent prevents float overflow during early optimization.
    pred_log = torch.clamp(pred_scaled * log_std + log_mean,
                           min=-20.0, max=20.0)
    true_log = target_scaled * log_std + log_mean
    pred_kwh = torch.expm1(pred_log)
    true_kwh = torch.expm1(true_log)

    # Divide by train-set kWh standard deviation to make the scale stable.
    pred_energy_scaled = (pred_kwh - energy_mean) / energy_std
    true_energy_scaled = (true_kwh - energy_mean) / energy_std
    kwh_mse = nn.functional.mse_loss(pred_energy_scaled, true_energy_scaled)
    return log_mse + lambda_kwh * kwh_mse, log_mse.detach(), kwh_mse.detach()


def predict_kwh(model, weather, static, log_mean, log_std, batch_size):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(weather), batch_size):
            end = min(start + batch_size, len(weather))
            xw = torch.as_tensor(weather[start:end], dtype=torch.float32, device=DEVICE)
            xs = torch.as_tensor(static[start:end], dtype=torch.float32, device=DEVICE)
            pred_scaled = model(xw, xs).detach().cpu().numpy().reshape(-1)
            pred_log = pred_scaled.astype(np.float64) * log_std + log_mean
            preds.append(np.expm1(pred_log))
    return np.concatenate(preds)


def save_checkpoint(path, model_state, epoch, val_loss, stats, args):
    data = {
        "model_state_dict": model_state,
        "best_epoch": epoch,
        "best_val_loss": float(val_loss),
        "weather_mean": stats["weather_mean"],
        "weather_std": stats["weather_std"],
        "static_mean": stats["static_mean"],
        "static_std": stats["static_std"],
        "target_mean": float(stats["log_mean"]),
        "target_std": float(stats["log_std"]),
        "target_transform": "log1p",
        "training_objective": "hybrid_log_mse_plus_standardized_kwh_mse",
        "lambda_kwh": float(args.lambda_kwh),
        "energy_train_mean": float(stats["energy_mean"]),
        "energy_train_std": float(stats["energy_std"]),
        "weather_features": 6,
        "static_features": len(STATIC_FEATURE_NAMES),
        "static_feature_names": STATIC_FEATURE_NAMES,
        "channels": CHANNELS,
        "kernel_size": KERNEL_SIZE,
        "dropout": DROPOUT,
        "random_state": args.split_seed,
        "training_seed": args.train_seed,
    }

    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(data, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    args = parse_args()
    set_seed(args.train_seed)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_dir = Path("runs") / config["run_name"]
    data_dir = run_dir / "tcn_data"
    if args.model_dir is None:
        raise ValueError(
            "Please provide --model-dir to avoid overwriting the original V3 model."
        )

    model_dir = Path(args.model_dir).resolve()

    protected_dir = (run_dir / "models_v3").resolve()

    if model_dir == protected_dir:
        raise ValueError(
            "This directory contains the original V3 model. "
            "Please choose a different --model-dir."
        )

    model_dir.mkdir(parents=True, exist_ok=True)

    # Same dataset as V1/V2; the building-group split depends on --split-seed.
    weather = np.load(data_dir / "X_weather.npy").astype(np.float32)
    static = np.load(data_dir / "X_static.npy").astype(np.float32)
    targets = np.load(data_dir / "y.npy").astype(np.float32)
    ids = np.load(data_dir / "ids.npy")
    group_ids = np.load(data_dir / "group_ids.npy")
    if not (len(weather) == len(static) == len(targets) == len(ids) == len(group_ids)):
        raise ValueError("Input array lengths do not match")
    if weather.shape[2] != 6 or static.shape[1] != len(STATIC_FEATURE_NAMES):
        raise ValueError("Unexpected weather/static feature dimensions")
    if np.any(targets <= 0) or not np.all(np.isfinite(targets)):
        raise ValueError("Annual-energy targets must be finite and positive")

    train_idx, val_idx, test_idx = split_indices(
        group_ids,
        config["surrogate"]["test_size"],
        config["surrogate"]["val_size"],
        random_state=args.split_seed
    )
    print(f"Device: {DEVICE}", flush=True)
    print(f"Groups train/val/test: {len(np.unique(group_ids[train_idx]))}/"
          f"{len(np.unique(group_ids[val_idx]))}/{len(np.unique(group_ids[test_idx]))}", flush=True)
    print(f"Samples train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}", flush=True)
    print(f"Hybrid loss: log MSE + {args.lambda_kwh} * standardized kWh MSE", flush=True)
    print(f"Output: {model_dir}", flush=True)

    weather_train, weather_val, weather_test, weather_mean, weather_std = standardize(
        weather[train_idx], weather[val_idx], weather[test_idx], axes=(0, 1)
    )
    static_train, static_val, static_test, static_mean, static_std = standardize(
        static[train_idx], static[val_idx], static[test_idx], axes=0
    )
    y_train, y_val, y_test = targets[train_idx], targets[val_idx], targets[test_idx]
    log_train = np.log1p(y_train)
    log_val = np.log1p(y_val)
    log_mean = float(log_train.mean())
    log_std = float(log_train.std())
    if log_std < 1e-8:
        log_std = 1.0
    energy_mean = float(y_train.mean())
    energy_std = float(y_train.std())
    if energy_std < 1e-8:
        energy_std = 1.0

    target_train = ((log_train - log_mean) / log_std).astype(np.float32)
    target_val = ((log_val - log_mean) / log_std).astype(np.float32)
    stats = {
        "weather_mean": weather_mean, "weather_std": weather_std,
        "static_mean": static_mean, "static_std": static_std,
        "log_mean": log_mean, "log_std": log_std,
        "energy_mean": energy_mean, "energy_std": energy_std,
    }

    xw_train = torch.from_numpy(np.ascontiguousarray(weather_train))
    xs_train = torch.from_numpy(np.ascontiguousarray(static_train))
    yt_train = torch.from_numpy(target_train)
    xw_val = torch.from_numpy(np.ascontiguousarray(weather_val))
    xs_val = torch.from_numpy(np.ascontiguousarray(static_val))
    yt_val = torch.from_numpy(target_val)

    model = TCNSurrogate(
        weather_features=6, static_features=len(STATIC_FEATURE_NAMES),
        channels=CHANNELS, kernel_size=KERNEL_SIZE, dropout=DROPOUT
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_loss, best_epoch, best_state = float("inf"), 0, None
    no_improvement = 0
    log_mean_tensor = torch.tensor(log_mean, device=DEVICE, dtype=torch.float32)
    log_std_tensor = torch.tensor(log_std, device=DEVICE, dtype=torch.float32)
    energy_mean_tensor = torch.tensor(energy_mean, device=DEVICE, dtype=torch.float32)
    energy_std_tensor = torch.tensor(energy_std, device=DEVICE, dtype=torch.float32)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_idx))
        total_train, total_train_log, total_train_kwh = 0.0, 0.0, 0.0
        for start in range(0, len(train_idx), args.batch_size):
            ix = perm[start:start + args.batch_size]
            xw = xw_train[ix].to(DEVICE)
            xs = xs_train[ix].to(DEVICE)
            yy = yt_train[ix].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xw, xs).reshape(-1)
            loss, log_part, kwh_part = hybrid_loss(
                pred, yy, log_mean_tensor, log_std_tensor,
                energy_mean_tensor, energy_std_tensor, args.lambda_kwh
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite train loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            n = len(ix)
            total_train += float(loss.item()) * n
            total_train_log += float(log_part.item()) * n
            total_train_kwh += float(kwh_part.item()) * n
        train_loss = total_train / len(train_idx)

        model.eval()
        total_val, total_val_log, total_val_kwh = 0.0, 0.0, 0.0
        with torch.no_grad():
            for start in range(0, len(val_idx), args.batch_size):
                end = min(start + args.batch_size, len(val_idx))
                xw = xw_val[start:end].to(DEVICE)
                xs = xs_val[start:end].to(DEVICE)
                yy = yt_val[start:end].to(DEVICE)
                pred = model(xw, xs).reshape(-1)
                loss, log_part, kwh_part = hybrid_loss(
                    pred, yy, log_mean_tensor, log_std_tensor,
                    energy_mean_tensor, energy_std_tensor, args.lambda_kwh
                )
                n = end - start
                total_val += float(loss.item()) * n
                total_val_log += float(log_part.item()) * n
                total_val_kwh += float(kwh_part.item()) * n
        val_loss = total_val / len(val_idx)
        if not np.isfinite(val_loss):
            raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(model_dir / "tcn_best_during_training.pt", best_state,
                            best_epoch, best_val_loss, stats, args)
            no_improvement = 0
        else:
            no_improvement += 1
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_log_mse": total_train_log / len(train_idx),
            "val_log_mse": total_val_log / len(val_idx),
            "train_kwh_mse_normalized": total_train_kwh / len(train_idx),
            "val_kwh_mse_normalized": total_val_kwh / len(val_idx),
            "best_epoch_so_far": best_epoch,
        })
        pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
        print(f"Epoch {epoch:3d} | Train loss: {train_loss:.6f} | "
              f"Val loss: {val_loss:.6f} | Best: {best_epoch}", flush=True)
        if no_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint")
    model.load_state_dict(best_state)
    save_checkpoint(model_dir / "tcn_surrogate_base.pt", best_state,
                    best_epoch, best_val_loss, stats, args)
    print(f"Best epoch: {best_epoch}; best validation loss: {best_val_loss:.6f}", flush=True)

    # Evaluate on all held-out test scenarios in physical kWh.
    predictions = predict_kwh(model, weather_test, static_test,
                              log_mean, log_std, args.batch_size)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    mape_pct = float(np.mean(np.abs(predictions - y_test) / y_test) * 100)
    r2 = float(r2_score(y_test, predictions))
    print(f"Test RMSE: {rmse:.2f} kWh", flush=True)
    print(f"Test MAE: {mae:.2f} kWh", flush=True)
    print(f"Test MAPE: {mape_pct:.4f}%", flush=True)
    print(f"Test R2: {r2:.6f}", flush=True)
    print(f"Saved model to: {model_dir / 'tcn_surrogate_base.pt'}", flush=True)

    # Export test predictions for error analysis and v1/v2/v3 comparison.
    original_csv = run_dir / "synthetic_dataset" / "dataset_valid.csv"
    original_df = pd.read_csv(original_csv)
    if not np.array_equal(original_df["id"].to_numpy(), ids):
        raise ValueError("Dataset rows do not match input arrays")
    results_df = original_df.iloc[test_idx].copy()
    results_df["predicted_energy"] = predictions
    results_df["absolute_error"] = np.abs(predictions - y_test)
    results_df["percentage_error"] = results_df["absolute_error"] / y_test * 100
    results_df["signed_error"] = predictions - y_test
    results_df.to_csv(model_dir / "tcn_v3_test_predictions.csv", index=False)
    with open(model_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "training_seed": args.train_seed,
            "split_seed": args.split_seed,
            "lambda_kwh": args.lambda_kwh,
            "test_samples": len(test_idx),
            "rmse_kwh": rmse,
            "mae_kwh": mae,
            "mape_percent": mape_pct,
            "r2": r2,
        }, f, indent=2)


if __name__ == "__main__":
    main()