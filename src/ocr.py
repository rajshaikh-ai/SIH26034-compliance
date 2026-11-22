"""OCR engine wrapper.

Primary backend: RapidOCR (PaddleOCR PP-OCR models on ONNX Runtime) - pure pip
install, no system binaries needed. Fallback: Tesseract via pytesseract when
the tesseract executable is installed.
"""
from __future__ import annotations

import logging
import os

import numpy as np


class OCREngine:
    def __init__(self, backend: str = "auto", threads: int | None = None):
        self.backend = None
        errors = []
        if backend in ("auto", "rapidocr"):
            try:
                logging.getLogger("RapidOCR").setLevel(logging.ERROR)
                from rapidocr import RapidOCR

                params = {"Global.log_level": "error"}
                if threads:
                    params["EngineConfig.onnxruntime.intra_op_num_threads"] = threads
                try:
                    self._rapid = RapidOCR(params=params)
                except Exception:
                    self._rapid = RapidOCR()
                logging.getLogger("RapidOCR").setLevel(logging.ERROR)
                self.backend = "rapidocr"
            except Exception as e:  # pragma: no cover
                errors.append(f"rapidocr: {e}")
        if self.backend is None and backend in ("auto", "tesseract"):
            try:
                import pytesseract

                if os.name == "nt" and os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
                    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                pytesseract.get_tesseract_version()
                self._tess = pytesseract
                self.backend = "tesseract"
            except Exception as e:  # pragma: no cover
                errors.append(f"tesseract: {e}")
        if self.backend is None:
            raise RuntimeError("No OCR backend available. Run: pip install rapidocr onnxruntime\n" + "\n".join(errors))

    # ------------------------------------------------------------------
    def read(self, img: np.ndarray) -> list[dict]:
        """Return text segments: text, conf, box (4 points), height, x0..y1."""
        segs = self._read_rapid(img) if self.backend == "rapidocr" else self._read_tesseract(img)
        return sorted(segs, key=lambda s: (s["cy"], s["x0"]))

    def _read_rapid(self, img: np.ndarray) -> list[dict]:
        res = self._rapid(img)
        if isinstance(res, tuple):  # legacy rapidocr_onnxruntime API
            items = res[0] or []
            boxes = [it[0] for it in items]
            txts = [it[1] for it in items]
            scores = [float(it[2]) for it in items]
        else:
            if res.boxes is None or res.txts is None:
                return []
            boxes, txts, scores = res.boxes, res.txts, res.scores
        out = []
        for box, txt, sc in zip(boxes, txts, scores):
            out.append(_segment(np.asarray(box, dtype=float), str(txt), float(sc)))
        return out

    def _read_tesseract(self, img: np.ndarray) -> list[dict]:
        d = self._tess.image_to_data(img, output_type=self._tess.Output.DICT, config="--psm 6")
        groups: dict = {}
        for i, word in enumerate(d["text"]):
            if not word.strip() or float(d["conf"][i]) < 0:
                continue
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            groups.setdefault(key, []).append(i)
        out = []
        for idxs in groups.values():
            x0 = min(d["left"][i] for i in idxs)
            y0 = min(d["top"][i] for i in idxs)
            x1 = max(d["left"][i] + d["width"][i] for i in idxs)
            y1 = max(d["top"][i] + d["height"][i] for i in idxs)
            box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
            text = " ".join(d["text"][i] for i in idxs)
            conf = float(np.mean([float(d["conf"][i]) for i in idxs])) / 100.0
            out.append(_segment(box, text, conf))
        return out


def _segment(box: np.ndarray, text: str, conf: float) -> dict:
    # rotation-invariant height: mean length of the two vertical edges
    h = (np.linalg.norm(box[3] - box[0]) + np.linalg.norm(box[2] - box[1])) / 2.0
    xs, ys = box[:, 0], box[:, 1]
    return dict(text=text, conf=conf, box=box.round(1).tolist(), height=float(h),
                x0=float(xs.min()), y0=float(ys.min()), x1=float(xs.max()), y1=float(ys.max()),
                cy=float(ys.mean()))


def group_lines(segs: list[dict], max_gap: float = 2.5) -> list[dict]:
    """Merge OCR segments that sit on the same visual line.

    A segment joins a line only if it is vertically aligned with the line's
    *first* segment (the band never grows) and horizontally close to it
    (gap < max_gap x text height). Text in another column of a multi-panel
    photo therefore stays a separate line."""
    lines: list[dict] = []
    for s in sorted(segs, key=lambda s: (s["cy"], s["x0"])):
        placed = False
        for ln in lines:
            ref = ln["segs"][0]
            overlap = min(ref["y1"], s["y1"]) - max(ref["y0"], s["y0"])
            if overlap <= 0.5 * min(ref["height"], s["height"]):
                continue
            if abs(ref["height"] - s["height"]) >= 0.5 * max(ref["height"], s["height"]):
                continue
            gap = max(s["x0"] - ln["x1"], ln["x0"] - s["x1"])
            if gap > max_gap * max(ref["height"], s["height"]):
                continue
            ln["segs"].append(s)
            ln["x0"], ln["x1"] = min(ln["x0"], s["x0"]), max(ln["x1"], s["x1"])
            ln["y0"], ln["y1"] = min(ln["y0"], s["y0"]), max(ln["y1"], s["y1"])
            placed = True
            break
        if not placed:
            lines.append(dict(segs=[s], x0=s["x0"], x1=s["x1"], y0=s["y0"], y1=s["y1"], height=s["height"]))
    out = []
    for ln in lines:
        segs_sorted = sorted(ln["segs"], key=lambda s: s["x0"])
        out.append(dict(
            text=" ".join(s["text"] for s in segs_sorted),
            conf=float(np.mean([s["conf"] for s in segs_sorted])),
            height=float(np.median([s["height"] for s in segs_sorted])),
            x0=min(s["x0"] for s in segs_sorted), x1=max(s["x1"] for s in segs_sorted),
            y0=ln["y0"], y1=ln["y1"], boxes=[s["box"] for s in segs_sorted],
        ))
    return sorted(out, key=lambda l: l["y0"])
