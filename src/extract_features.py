"""Batch feature extraction (parallel OCR) for a labelled image dataset.

Input : a folder with labels.csv (image_id, filename, compliant, violations ...)
Output: data/processed/features.csv  +  data/processed/ocr/<image_id>.json

Works for the synthetic set or any real dataset in the same format.
Usage:  python -m src.extract_features [--data-dir data/synthetic] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src import config as C

_ANALYZER = None


def _init_worker(threads: int) -> None:
    global _ANALYZER
    from src.features import LabelAnalyzer

    _ANALYZER = LabelAnalyzer(threads=threads)


def _process(args: tuple) -> dict:
    image_id, path, panel_mm = args
    try:
        res = _ANALYZER.analyze(path, panel_width_mm=panel_mm)
    except Exception as e:  # keep going on corrupt files
        return dict(image_id=image_id, error=str(e))
    ocr_dump = [dict(text=l["text"], conf=round(l["conf"], 3), height=round(l["height"], 1), boxes=l["boxes"])
                for l in res["lines"]]
    with open(C.OCR_CACHE_DIR / f"{image_id}.json", "w", encoding="utf-8") as fh:
        json.dump(ocr_dump, fh, ensure_ascii=False)
    return dict(image_id=image_id, **res["features"], ocr_text=res["text"].replace("\n", " | "),
                rule_compliant=int(res["rule"]["compliant"]), rule_violations="|".join(res["rule"]["violations"]))


def extract(data_dir: Path, workers: int, panel_mm: float = C.PANEL_WIDTH_MM) -> pd.DataFrame:
    C.ensure_dirs()
    labels = normalise_labels(pd.read_csv(data_dir / "labels.csv"))
    if "panel_width_mm" not in labels:
        labels["panel_width_mm"] = panel_mm
    jobs = [(r.image_id, str(data_dir / r.filename), float(r.panel_width_mm)) for r in labels.itertuples()]
    threads = max(1, (os.cpu_count() or 2) // max(workers, 1))
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(threads,)) as ex:
        futs = [ex.submit(_process, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 50 == 0 or i == len(jobs):
                rate = i / (time.time() - t0)
                print(f"  OCR {i}/{len(jobs)}  ({rate:.1f} img/s, ETA {(len(jobs) - i) / rate:.0f}s)")
    feats = pd.DataFrame(rows)
    errs = feats[feats.get("error").notna()] if "error" in feats else feats.iloc[0:0]
    if len(errs):
        print(f"  WARNING: {len(errs)} images failed: {errs.image_id.tolist()[:5]}")
        feats = feats[feats["error"].isna()].drop(columns="error")
    df = labels.merge(feats, on="image_id", how="inner").sort_values("image_id")
    df.to_csv(C.FEATURES_CSV, index=False)
    return df


def normalise_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Country of origin is only mandatory for imported goods: drop that
    violation from domestic labels (keeps older generated sets consistent)."""
    if "imported" not in labels:
        return labels
    v = labels["violations"].fillna("").astype(str)
    dom = labels["imported"] == 0
    v = v.where(~dom, v.apply(lambda s: "|".join(x for x in s.split("|") if x and x != "MISSING_COUNTRY_OF_ORIGIN")))
    labels = labels.copy()
    labels["violations"] = v
    labels["n_violations"] = v.apply(lambda s: len([x for x in s.split("|") if x]))
    labels["compliant"] = (labels["n_violations"] == 0).astype(int)
    return labels


def rebuild_from_cache(data_dir: Path) -> pd.DataFrame:
    """Recompute features from cached OCR results (no OCR needed) - used after
    changing the rule engine or feature definitions."""
    from src.features import features_from_lines, lines_from_cache
    from src.preprocess import estimate_skew, load_image
    import cv2

    labels = normalise_labels(pd.read_csv(data_dir / "labels.csv"))
    labels.to_csv(data_dir / "labels.csv", index=False)
    rows = []
    for r in labels.itertuples():
        cache = C.OCR_CACHE_DIR / f"{r.image_id}.json"
        if not cache.exists():
            continue
        bgr = load_image(data_dir / r.filename)
        lines = lines_from_cache(json.loads(cache.read_text(encoding="utf-8")))
        skew = estimate_skew(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        res = features_from_lines(lines, bgr, getattr(r, "panel_width_mm", C.PANEL_WIDTH_MM), skew)
        rows.append(dict(image_id=r.image_id, **res["features"], ocr_text=res["text"].replace("\n", " | "),
                         rule_compliant=int(res["rule"]["compliant"]),
                         rule_violations="|".join(res["rule"]["violations"])))
    df = labels.merge(pd.DataFrame(rows), on="image_id", how="inner").sort_values("image_id")
    df.to_csv(C.FEATURES_CSV, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=C.DATASET_DIR)
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--panel-mm", type=float, default=C.PANEL_WIDTH_MM)
    ap.add_argument("--from-cache", action="store_true", help="rebuild features from cached OCR (fast)")
    a = ap.parse_args()
    df = rebuild_from_cache(a.data_dir) if a.from_cache else extract(a.data_dir, a.workers, a.panel_mm)
    print(f"Saved features for {len(df)} images to {C.FEATURES_CSV}")


if __name__ == "__main__":
    main()
