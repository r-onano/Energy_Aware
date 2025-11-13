import argparse
import random
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

"""
Creates a tiny synthetic dataset (images + frame_index.parquet) that the
existing pipeline can consume end-to-end without external downloads.

- 60 frames, split 40/10/10
- Simple boxes (rectangles) drawn on random backgrounds
- "object_density" etc. seeded into the index to mimic labels
- Images saved under data/synth/images
"""

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
        color = (int(rng.randint(0,255)), int(rng.randint(0,255)), int(rng.randint(0,255)))
        cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness=2)
    return img

def main(args):
    root = Path("data/synth/images")
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    n = args.num_frames
    rng = random.Random(42)

    # Create images and synth “labels” as index columns
    for i in range(n):
        # vary object counts + brightness profile
        obj = rng.choice([0, 1, 2, 3, 5, 8, 12])
        img = make_img(n_boxes=obj, seed=1000+i)

        img_path = root / f"frame_{i:05d}.jpg"
        cv2.imwrite(str(img_path), img)

        # proxies
        small_ratio = 0.3 if obj >= 5 else 0.1
        occ_proxy = 0.2 if obj >= 8 else 0.05

        rows.append({
            "dataset": "synth",
            "frame_id": f"synth_{i:05d}",
            "img_path": str(img_path),
            "label_path": "",
            "object_density": obj,
            "small_obj_ratio": small_ratio,
            "occlusion_proxy": occ_proxy,
        })

    df = pd.DataFrame(rows)

    # Simple split: 40/10/10
    df["split"] = "train"
    df.loc[int(0.66*n):int(0.83*n), "split"] = "val"
    df.loc[int(0.83*n):, "split"] = "test"

    out = Path("results/frame_index.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Synth demo ready → {out} (rows={len(df)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-frames", type=int, default=60)
    main(ap.parse_args())
