"""End-to-end pipeline: dataset -> OCR features -> training/evaluation -> figures.

    python run_pipeline.py                 # full run (1200 synthetic labels)
    python run_pipeline.py --n 400         # quicker run
    python run_pipeline.py --skip-generate # reuse existing images
    python run_pipeline.py --skip-ocr      # reuse existing features.csv (retrain only)
"""
import argparse
import os
import time

from src import config as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-ocr", action="store_true")
    a = ap.parse_args()
    C.ensure_dirs()
    t0 = time.time()

    if not (a.skip_generate or a.skip_ocr):
        print(f"[1/3] Generating {a.n} synthetic label images ...")
        from src.generate_dataset import generate
        generate(a.n)
    if not a.skip_ocr:
        print(f"[2/3] Preprocessing + OCR + feature extraction ({a.workers} workers) ...")
        from src.extract_features import extract
        extract(C.DATASET_DIR, a.workers)
    print("[3/3] Training, evaluating and plotting ...")
    from src import train
    train.main()
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Launch the dashboard with:  streamlit run app.py")


if __name__ == "__main__":
    main()
