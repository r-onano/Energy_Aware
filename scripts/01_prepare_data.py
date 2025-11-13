import argparse
from pathlib import Path
import pandas as pd

from src.common.io import read_yaml, write_parquet
from src.common.log import info, ok, warn
from src.common.seed import set_all_seeds
from src.data.kitti import build_kitti_index
from src.data.bdd import build_bdd_index
from src.data.sampler import temporal_split


def main(args):
    cfg = read_yaml(args.config)
    set_all_seeds(cfg.get("index", {}).get("random_seed", 42))

    out_path = Path(cfg["index"]["out_path"])

    frames = []

    # KITTI
    kitti_img = Path(cfg["kitti"]["images_dir"])
    kitti_lbl = Path(cfg["kitti"]["labels_dir"])
    kitti_ext = cfg["kitti"].get("img_ext", ".png")
    if kitti_img.exists():
        info(f"Indexing KITTI from {kitti_img}")
        df_kitti = build_kitti_index(kitti_img, kitti_lbl, kitti_ext)
        frames.append(df_kitti)
    else:
        warn("KITTI images_dir not found; skipping.")

    # BDD
    bdd_img = Path(cfg["bdd"]["images_dir"])
    bdd_lbl = Path(cfg["bdd"]["labels_coco"])
    bdd_ext = cfg["bdd"].get("img_ext", ".jpg")
    if bdd_img.exists():
        info(f"Indexing BDD from {bdd_img}")
        df_bdd = build_bdd_index(bdd_img, bdd_lbl, bdd_ext)
        frames.append(df_bdd)
    else:
        warn("BDD images_dir not found; skipping.")

    if not frames:
        raise SystemExit("No datasets were indexed. Please check paths in configs/data.yaml")

    df = pd.concat(frames, ignore_index=True)
    df = temporal_split(
        df,
        train_ratio=cfg["index"]["train_ratio"],
        val_ratio=cfg["index"]["val_ratio"],
        test_ratio=cfg["index"]["test_ratio"],
        seed=cfg["index"]["random_seed"],
        shuffle=cfg["index"]["shuffle"],
    )

    write_parquet(df, out_path)
    ok(f"Wrote frame index: {out_path}  (rows={len(df)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    main(parser.parse_args())
