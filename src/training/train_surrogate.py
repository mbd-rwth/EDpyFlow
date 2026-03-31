"""
Step 5 — Surrogate Model Training

Trains an XGBoost surrogate model (EDSurrogate) on the synthetic dataset
produced in Step 4 and evaluates its performance on a held-out test set.

Reads:  runs/{run_name}/synthetic_dataset/dataset.csv, config.yaml
Output: runs/{run_name}/models/{model_name}.json
"""

from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import xgboost as xgb
import numpy as np
import os
import yaml


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

run_name      = config["run_name"]
surrogate_cfg = config["surrogate"]
model_name    = surrogate_cfg["model_name"]

run_dir = os.path.join("runs", run_name)
dataset_path = os.path.join(run_dir, "synthetic_dataset", "dataset.csv")
model_dir = os.path.join(run_dir, "models")

df = pd.read_csv(dataset_path, index_col="id")

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=["building_type", "location", "refurbishment_status"])

X = df_encoded.drop(columns=["total_energy"])
y = df_encoded["total_energy"]

# Train / validation / test split
val_and_test_size = surrogate_cfg["val_size"] + surrogate_cfg["test_size"]
relative_test_size = surrogate_cfg["test_size"] / val_and_test_size

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=val_and_test_size, random_state=surrogate_cfg["random_state"], shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=relative_test_size, random_state=surrogate_cfg["random_state"], shuffle=True
)

model = xgb.XGBRegressor(
    n_estimators=surrogate_cfg["n_estimators"],
    learning_rate=surrogate_cfg["learning_rate"],
    early_stopping_rounds=surrogate_cfg["early_stopping_rounds"],
    random_state=surrogate_cfg["random_state"],
    verbosity=1,
    objective="reg:squarederror",
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

y_pred = model.predict(X_test)

print(f"Best iteration:  {model.best_iteration}")
print(f"Test set size:   {len(X_test)}")
print(f"MSE:             {metrics.mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE:            {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE:             {metrics.mean_absolute_error(y_test, y_pred):.4f}")
print(f"MAPE:            {metrics.mean_absolute_percentage_error(y_test, y_pred):.4f}")
print(f"R²:              {metrics.r2_score(y_test, y_pred):.4f}")

os.makedirs(model_dir, exist_ok=True)
model.save_model(os.path.join(model_dir, f"{model_name}.json"))
print(f"Saved model to {model_dir}/{model_name}.json")

test_set_path = os.path.join(run_dir, "synthetic_dataset", "test_set.csv")
df.loc[X_test.index].to_csv(test_set_path)
print(f"Saved test set  to {test_set_path}")