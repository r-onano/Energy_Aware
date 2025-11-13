import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False


def make_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    return None


def build_regressor(cfg: Dict):
    name = cfg["model"]["name"]
    if name == "rf":
        p = cfg["model"]["rf"]
        return RandomForestRegressor(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth", None),
            random_state=p.get("random_state", 42),
            n_jobs=-1,
        )
    if name == "lgbm" and _HAS_LGBM:
        p = cfg["model"]["lgbm"]
        return lgb.LGBMRegressor(
            n_estimators=p.get("n_estimators", 500),
            learning_rate=p.get("learning_rate", 0.05),
            num_leaves=p.get("num_leaves", 31),
            subsample=0.9, colsample_bytree=0.9, random_state=42
        )
    if name == "mlp":
        p = cfg["model"]["mlp"]
        return MLPRegressor(
            hidden_layer_sizes=tuple(p.get("hidden_layers", [128, 64])),
            activation=p.get("activation", "relu"),
            learning_rate_init=p.get("lr", 1e-3),
            max_iter=p.get("epochs", 30),
            random_state=42
        )
    if name == "linear":
        return LinearRegression()
    raise ValueError(f"Unknown model name: {name}")


def load_feature_matrix(features_path: Path, feat_cols: list, split_filter=None) -> pd.DataFrame:
    df = pd.read_parquet(features_path)
    if split_filter:
        df = df[df["split"].isin(split_filter)].copy()
    # Drop rows with NaNs in features
    df = df.dropna(subset=feat_cols)
    return df


def maybe_load_latency_labels(baseline_dir: Path) -> pd.DataFrame | None:
    """Expect per-frame latency CSV produced by 05_run_perception_baseline.py"""
    lat_csv = baseline_dir / "per_frame.csv"
    if lat_csv.exists():
        lat = pd.read_csv(lat_csv)
        # keep frame_id + latency_ms, power_w
        return lat[["frame_id", "latency_ms", "power_w"]]
    return None


def simulate_targets_from_features(df: pd.DataFrame, feat_cols: list) -> pd.Series:
    """Heuristic fallback target if baseline latency not available."""
    # Use object density and motion as primary cost drivers; illumination for minor effect.
    od = df.get("object_density", 0).astype(float)
    mm = df.get("motion_mag_p90", 0).astype(float)
    br = df.get("brightness_std", 0).astype(float)
    # ms in [4, 30] roughly
    t = 4.0 + 0.9 * np.tanh(0.15 * od) * 10.0 + 0.6 * np.tanh(0.8 * mm) * 10.0 + 0.2 * np.tanh(0.1 * br) * 5.0
    noise = np.random.RandomState(42).normal(0, 0.8, size=len(df))
    return t + noise


def train_regressor(cfg: Dict, features_path: Path, baseline_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_cols = cfg["features"]["cols"]
    target_name = cfg["target"]["name"]
    scaler_kind = cfg["target"].get("scaler", "standard")

    # Load features for train/val/test according to 'split' col
    df_all = load_feature_matrix(features_path, feat_cols, split_filter=["train", "val", "test"])

    # Labels: prefer baseline per-frame latency, else simulate
    labels = maybe_load_latency_labels(baseline_dir)
    if labels is not None and target_name == "compute_time_ms":
        df = df_all.merge(labels.rename(columns={"latency_ms": "compute_time_ms"}), on="frame_id", how="left")
    elif labels is not None and target_name == "gpu_power_w":
        df = df_all.merge(labels[["frame_id", "power_w"]].rename(columns={"power_w": "gpu_power_w"}), on="frame_id", how="left")
    else:
        df = df_all.copy()
        df[target_name] = simulate_targets_from_features(df_all, feat_cols)

    # Split views
    train_df = df[df["split"] == "train"].dropna(subset=[target_name]).copy()
    val_df = df[df["split"] == "val"].dropna(subset=[target_name]).copy()
    test_df = df[df["split"] == "test"].dropna(subset=[target_name]).copy()

    X_train, y_train = train_df[feat_cols].values, train_df[target_name].values
    X_val, y_val = val_df[feat_cols].values, val_df[target_name].values
    X_test, y_test = test_df[feat_cols].values, test_df[target_name].values

    scaler = make_scaler(scaler_kind)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    model = build_regressor(cfg)
    model.fit(X_train, y_train)

    # Simple eval
    def _metrics(X, y):
        pred = model.predict(X)
        mae = float(np.mean(np.abs(pred - y)))
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        return {"mae": mae, "rmse": rmse}

    m_train = _metrics(X_train, y_train)
    m_val = _metrics(X_val, y_val)
    m_test = _metrics(X_test, y_test)

    # Save
    joblib.dump(model, out_dir / "scheduler_model.pkl")
    if scaler is not None:
        joblib.dump(scaler, out_dir / "feature_scaler.pkl")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"train": m_train, "val": m_val, "test": m_test, "target": target_name}, f, indent=2)

    return m_train, m_val, m_test
