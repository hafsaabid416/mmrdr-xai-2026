#Resize all CFP images once to 512x512 and save them as a JPEG cache.
#Illumination normalisation is applied at load time so it can be switched on or off per experiment.

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmrdr.preprocessing import preprocess  


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="MMRDR-CFP folder (contains FP.csv and img/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    images = pd.read_csv(src / "FP.csv")["image"].tolist()

    def work(rel: str) -> int:
        dst = out / Path(rel).name
        if dst.exists():
            return 0
        img = cv2.imread(str(src / rel), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(src / rel)
        cv2.imwrite(str(dst), preprocess(img, a.size), [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        return 1

    with ThreadPoolExecutor(a.workers) as ex:
        done = sum(ex.map(work, images))
    print(f"wrote {done} new images; cache has {len(list(out.glob('*.jpg')))} / {len(images)}")


if __name__ == "__main__":
    main()
