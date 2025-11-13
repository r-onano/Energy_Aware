import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.common.io import read_yaml, read_image, write_parquet
from src.common.log import info, ok, warn
from src.features.compose import build_feature_pipeline


def main(args):
    data_cfg = read_yaml(args.data_config)
    feat_cfg = read_yaml(args.feat_config)

    index_path = Path(data_cfg["index"]["out_path"])
    out_path = Path(feat_cfg["io"]["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not index_path.exists():
        raise SystemExit(f"Frame index not found: {index_path}. Run scripts/01_prepare_data.py first.")

    df = pd.read_parquet(index_path)

    # Optional subsampling for quick iterations
    if args.stride and args.stride > 1:
        df = df.iloc[:: args.stride].copy()
        info(f"Subsampled with stride={args.stride}: {len(df)} rows")

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
        info(f"Limited to first {args.limit} rows")

    pipe = build_feature_pipeline(feat_cfg)

    rows = []
    prev_dataset = None
    errors = 0

    it = tqdm(df.itertuples(index=False), total=len(df), desc="Extracting features")
    for row in it:
        row_dict = row._asdict() if hasattr(row, "_asdict") else dict(row)

        # Optional: reset flow when switching dataset (to avoid cross-dataset flow)
        if args.reset_per_dataset and prev_dataset is not None and row_dict["dataset"] != prev_dataset and getattr(pipe, "flow", None) is not None:
            pipe = build_feature_pipeline(feat_cfg)  # reset stateful flow
        prev_dataset = row_dict["dataset"]

        img = read_image(row_dict["img_path"])
        if img is None:
            warn(f"Could not read image: {row_dict['img_path']} (skipping)")
            errors += 1
            continue

        feats = pipe.compute(img, row_dict)
        feats.update(
            dict(
                dataset=row_dict["dataset"],
                frame_id=row_dict["frame_id"],
                split=row_dict["split"],
                img_path=row_dict["img_path"],
            )
        )
        rows.append(feats)

    if not rows:
        raise SystemExit("No features extracted. Check your image paths and configs.")

    feat_df = pd.DataFrame(rows)

    # Column order (nice-to-have)
    first_cols = ["dataset", "split", "frame_id", "img_path"]
    other_cols = [c for c in feat_df.columns if c not in first_cols]
    feat_df = feat_df[first_cols + other_cols]

    write_parquet(feat_df, out_path)
    suffix = f" (skipped {errors} unreadable images)" if errors else ""
    ok(f"Wrote features: {out_path}  (rows={len(feat_df)}, cols={len(feat_df.columns)}){suffix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=str, required=True)
    parser.add_argument("--feat-config", type=str, required=True)
    parser.add_argument("--stride", type=int, default=1, help="Take every Nth frame (default: 1 = all)")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N frames (0 = all)")
    parser.add_argument("--reset-per-dataset", action="store_true", default=True,
                        help="Reset motion tracker when dataset changes (default: on)")
    parser.add_argument("--no-reset-per-dataset", dest="reset_per_dataset", action="store_false",
                        help="Disable per-dataset motion reset")
    args = parser.parse_args()
    main(args)
