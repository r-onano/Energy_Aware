import argparse
from pathlib import Path
import pandas as pd

from src.common.io import read_yaml, write_parquet
from src.common.log import info, ok, warn
from src.common.seed import set_all_seeds
from src.data.kitti import build_kitti_index
from src.data.bdd import build_bdd_index
from src.data.sampler import temporal_split


def _summarize_df(df: pd.DataFrame, name: str):
    if df is None or df.empty:
        warn(f"{name}: 0 frames")
        return
    n_total = len(df)
    n_has_lbl = int(df["label_path"].astype(str).ne("").sum()) if "label_path" in df.columns else 0
    info(f"{name}: {n_total} frames (with labels: {n_has_lbl})")


def main(args):
    cfg = read_yaml(args.config)
    idx_cfg = cfg.get("index", {})
    set_all_seeds(idx_cfg.get("random_seed", 42))

    out_path = Path(idx_cfg["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []

    # -----------------------
    # KITTI
    # -----------------------
    kitti_img = Path(cfg["kitti"]["images_dir"])
    kitti_lbl = Path(cfg["kitti"]["labels_dir"])
    kitti_ext = args.kitti_ext or cfg["kitti"].get("img_ext", ".png")

    if kitti_img.exists():
        info(f"Indexing KITTI from {kitti_img}")
        df_kitti = build_kitti_index(kitti_img, kitti_lbl, kitti_ext)

        if not args.allow_unlabeled and "label_path" in df_kitti.columns:
            before = len(df_kitti)
            df_kitti = df_kitti[df_kitti["label_path"].astype(str) != ""].copy()
            removed = before - len(df_kitti)
            if removed:
                warn(f"KITTI: removed {removed} unlabeled frames (use --allow-unlabeled to keep).")

        _summarize_df(df_kitti, "KITTI")
        frames.append(df_kitti)
    else:
        warn("KITTI images_dir not found; skipping.")

    # -----------------------
    # BDD100K
    # -----------------------
    bdd_img = Path(cfg["bdd"]["images_dir"])
    bdd_lbl = Path(cfg["bdd"]["labels_coco"])
    bdd_ext = args.bdd_ext or cfg["bdd"].get("img_ext", ".jpg")

    if bdd_img.exists():
        info(f"Indexing BDD from {bdd_img}")
        try:
            df_bdd = build_bdd_index(bdd_img, bdd_lbl, bdd_ext)
        except Exception as e:
            warn(f"BDD indexing failed ({e}); skipping BDD.")
            df_bdd = pd.DataFrame()

        if not args.allow_unlabeled and "label_path" in df_bdd.columns:
            before = len(df_bdd)
            df_bdd = df_bdd[df_bdd["label_path"].astype(str) != ""].copy()
            removed = before - len(df_bdd)
            if removed:
                warn(f"BDD: removed {removed} unlabeled frames (use --allow-unlabeled to keep).")

        _summarize_df(df_bdd, "BDD")
        if not df_bdd.empty:
            frames.append(df_bdd)
    else:
        warn("BDD images_dir not found; skipping.")

    # -----------------------
    # Combine & optional subsample
    # -----------------------
    if not frames:
        raise SystemExit("No datasets were indexed. Check paths in configs/data.yaml")

    df = pd.concat(frames, ignore_index=True)

    if args.stride and args.stride > 1:
        df = df.iloc[:: args.stride].copy()
        info(f"Applied stride={args.stride}; rows → {len(df)}")
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
        info(f"Applied limit={args.limit}; rows → {len(df)}")

    # -----------------------
    # Split
    # -----------------------
    df = temporal_split(
        df,
        train_ratio=idx_cfg["train_ratio"],
        val_ratio=idx_cfg["val_ratio"],
        test_ratio=idx_cfg["test_ratio"],
        seed=idx_cfg["random_seed"],
        shuffle=idx_cfg["shuffle"],
    )

    # -----------------------
    # Save
    # -----------------------
    write_parquet(df, out_path)
    ok(f"Wrote frame index: {out_path}  (rows={len(df)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--allow-unlabeled", action="store_true", default=False,
                        help="Keep frames without labels (default: drop).")
    parser.add_argument("--limit", type=int, default=0, help="Limit total rows before split (0 = no limit).")
    parser.add_argument("--stride", type=int, default=1, help="Take every Nth row before split (default: 1).")
    parser.add_argument("--kitti-ext", type=str, default=None, help="Override KITTI image extension (e.g., .png/.jpg).")
    parser.add_argument("--bdd-ext", type=str, default=None, help="Override BDD image extension (e.g., .jpg).")
    main(parser.parse_args())
