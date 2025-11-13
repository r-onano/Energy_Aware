import argparse
import random
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

from src.common.io import read_yaml, write_parquet
from src.common.seed import set_all_seeds


def make_img(w=640, h=360, n_boxes=0, seed=0):
    rng = np.random.RandomState(seed)
    img = rng.randint(20, 230, size=(h, w, 3), dtype=np.uint8)
    # optional global brightness shift
    if rng.rand() < 0.3:
        img = np.clip(img + rng.randint(-40, 40), 0, 255).astype(np.uint8)
    # draw boxes
    for _ in range(n_boxes):
        x1 = rng.randint(0, w - 40)
        y1 = rng.randint(0, h - 40)
        x2 = int(np.clip(x1 + rng.randint(20, 120), 0, w - 1))
        y2 = int(np.clip(y1 + rng.randint(20, 120), 0, h - 1))
        color = (int(rng.randint(0, 255)), int(rng.randint(0, 255)), int(rng.randint(0, 255)))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
    return img


def main(args):
    # resolve index out path
    if args.data_config:
        cfg = read_yaml(args.data_config)
        out_index = Path(cfg["index"]["out_path"])
    else:
        out_index = Path(args.out_index)

    set_all_seeds(args.seed)

    img_root = Path(args.img_dir)
    img_root.mkdir(parents=True, exist_ok=True)
    out_index.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    n = int(args.num_frames)
    rng = random.Random(args.seed)

    for i in range(n):
        # vary object counts + brightness profile
        obj = rng.choice([0, 1, 2, 3, 5, 8, 12])
        img = make_img(n_boxes=obj, seed=1000 + i)

        img_path = img_root / f"frame_{i:05d}.jpg"
        ok = cv2.imwrite(str(img_path), img)
        if not ok:
            raise SystemExit(f"Failed to write image: {img_path}")

        # simple proxies to mimic labels-derived features
        small_ratio = 0.3 if obj >= 5 else 0.1
        occ_proxy = 0.2 if obj >= 8 else 0.05

        rows.append({
            "dataset": "synth",
            "frame_id": f"synth_{i:05d}",
            "img_path": str(img_path),
            "label_path": "",              # no labels for synth
            "object_density": obj,
            "small_obj_ratio": small_ratio,
            "occlusion_proxy": occ_proxy,
        })

    df = pd.DataFrame(rows)

    # Split: 66/17/17 ~ (40/10/10 for 60 frames)
    df["split"] = "train"
    df.loc[int(0.66 * n):int(0.83 * n), "split"] = "val"
    df.loc[int(0.83 * n):, "split"] = "test"

    write_parquet(df, out_index)
    print(f"Synth demo ready → {out_index} (rows={len(df)}). "
          f"Images in: {img_root}  | splits: "
          f"train={sum(df.split=='train')}, val={sum(df.split=='val')}, test={sum(df.split=='test')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", type=str, default="configs/data.yaml",
                    help="Config file to read index.out_path from (default: configs/data.yaml)")
    ap.add_argument("--out-index", type=str, default="results/frame_index.parquet",
                    help="Index parquet path if --data-config is not provided/used")
    ap.add_argument("--img-dir", type=str, default="data/synth/images",
                    help="Where to write synthetic images")
    ap.add_argument("--num-frames", type=int, default=60,
                    help="How many frames to synthesize")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
    args = ap.parse_args()
    main(args)
