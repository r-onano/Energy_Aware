import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.common.io import read_yaml, read_image, write_parquet
from src.common.log import info, ok
from src.features.compose import build_feature_pipeline


def main(args):
    data_cfg = read_yaml(args.data_config)
    feat_cfg = read_yaml(args.feat_config)

    index_path = Path(data_cfg["index"]["out_path"])
    out_path = Path(feat_cfg["io"]["out_path"])

    df = pd.read_parquet(index_path)

    pipe = build_feature_pipeline(feat_cfg)

    rows = []
    prev_dataset = None
    # motion tracker resets automatically per pipeline instance; if you want per-dataset reset:
    # just reinstantiate when dataset changes.
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        img = read_image(row["img_path"])

        # optional: reset flow when switching dataset (to avoid cross-dataset flow)
        if prev_dataset is not None and row["dataset"] != prev_dataset and pipe.flow is not None:
            pipe = build_feature_pipeline(feat_cfg)  # reset stateful flow
        prev_dataset = row["dataset"]

        feats = pipe.compute(img, dict(row))
        feats.update(
            dict(
                dataset=row["dataset"],
                frame_id=row["frame_id"],
                split=row["split"],
                img_path=row["img_path"],
            )
        )
        rows.append(feats)

    feat_df = pd.DataFrame(rows)

    # Column order (nice-to-have)
    first_cols = ["dataset", "split", "frame_id", "img_path"]
    other_cols = [c for c in feat_df.columns if c not in first_cols]
    feat_df = feat_df[first_cols + other_cols]

    write_parquet(feat_df, out_path)
    ok(f"Wrote features: {out_path}  (rows={len(feat_df)}, cols={len(feat_df.columns)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=str, required=True)
    parser.add_argument("--feat-config", type=str, required=True)
    main(parser.parse_args())
