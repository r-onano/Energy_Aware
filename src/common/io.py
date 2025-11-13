import os
import yaml
from pathlib import Path
import pandas as pd
import cv2


def read_yaml(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | os.PathLike):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: str | os.PathLike):
    ensure_dir(Path(path).parent)
    df.to_parquet(path, index=False)


def read_image(path: str):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img
