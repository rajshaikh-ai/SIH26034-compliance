"""Feature extraction: image -> preprocessing -> OCR -> rule checks -> feature vector.

`LabelAnalyzer.analyze()` is the single entry point used by the batch feature
extractor, the CLI predictor and the Streamlit app, so training and inference
always see identical features."""
from __future__ import annotations

import numpy as np

from src import config as C
from src.ocr import OCREngine, group_lines
from src.preprocess import image_quality_features, load_image, preprocess
from src.rules import FIELDS, check_label, safe_log

NUMERIC_FEATURES = (
    [f"rule_{f}" for f in FIELDS]
    + [f"fuzzy_{f}" for f in FIELDS]
    + [
        "netqty_value_log", "netqty_base_log", "netqty_unit_standard", "netqty_unit_nonstandard",
        "netqty_numeral_mm", "netqty_required_mm", "netqty_height_ratio", "netqty_line_height_rel",
        "mrp_price_log", "has_currency", "has_pin", "has_phone", "has_email", "n_dates",
        "ocr_n_lines", "ocr_n_chars", "ocr_mean_conf", "ocr_min_conf", "ocr_digit_ratio", "ocr_upper_ratio",
        "ocr_text_area", "ocr_median_height_mm",
        "img_sharpness", "img_contrast", "img_brightness", "img_edge_density", "img_noise", "img_aspect",
        "skew_angle_abs", "is_imported",
    ]
    # stacked rule-engine findings (one flag per violation)
    + [f"rv_{v}" for v in C.VIOLATIONS]
)
TEXT_FEATURE = "ocr_text"

# Photo-quality / OCR-statistics features. In the training data image quality is
# independent of compliance by design, so these carry no real signal - but on real
# phone photos (darker, tilted, lower OCR confidence) they lie far outside the
# training range and make classifiers extrapolate. They are still computed and
# shown in the app, just not fed to the models.
NUISANCE_FEATURES = {
    "ocr_n_lines", "ocr_n_chars", "ocr_mean_conf", "ocr_min_conf", "ocr_digit_ratio", "ocr_upper_ratio",
    "ocr_text_area", "ocr_median_height_mm", "img_sharpness", "img_contrast", "img_brightness",
    "img_edge_density", "img_noise", "img_aspect", "skew_angle_abs",
}
MODEL_FEATURES = [f for f in NUMERIC_FEATURES if f not in NUISANCE_FEATURES]


def build_features(lines: list[dict], rule: dict, quality: dict, img_shape, px_per_mm: float, skew: float) -> dict:
    F = rule["fields"]
    feats: dict = {}
    for f in FIELDS:
        feats[f"rule_{f}"] = float(F[f]["found"])
        feats[f"fuzzy_{f}"] = float(F[f]["fuzzy"])

    nq = F["net_qty"]
    heights = [l["height"] for l in lines] or [0.0]
    med_h = float(np.median(heights)) if lines else 0.0
    if nq["found"]:
        q = nq["qty"]
        feats.update(
            netqty_value_log=safe_log(q["value"]), netqty_base_log=safe_log(q["base"]),
            netqty_unit_standard=float(q["standard"]), netqty_unit_nonstandard=float(not q["standard"]),
            netqty_numeral_mm=nq["numeral_mm"], netqty_required_mm=nq["required_mm"],
            netqty_height_ratio=nq["numeral_mm"] / nq["required_mm"],
            netqty_line_height_rel=lines[nq["line"]]["height"] / med_h if med_h else 0.0,
        )
    else:
        feats.update(netqty_value_log=0.0, netqty_base_log=0.0, netqty_unit_standard=0.0,
                     netqty_unit_nonstandard=0.0, netqty_numeral_mm=0.0, netqty_required_mm=0.0,
                     netqty_height_ratio=0.0, netqty_line_height_rel=0.0)

    feats["mrp_price_log"] = safe_log(F["mrp"].get("price", 0.0)) if F["mrp"]["found"] else 0.0
    for k in ("has_currency", "has_pin", "has_phone", "has_email"):
        feats[k] = float(rule[k])
    feats["n_dates"] = float(rule["n_dates"])

    text = rule["full_text"]
    chars = [c for c in text if not c.isspace()]
    H, W = img_shape[:2]
    area = sum((l["x1"] - l["x0"]) * (l["y1"] - l["y0"]) for l in lines)
    feats.update(
        ocr_n_lines=float(len(lines)), ocr_n_chars=float(len(chars)),
        ocr_mean_conf=float(np.mean([l["conf"] for l in lines])) if lines else 0.0,
        ocr_min_conf=float(np.min([l["conf"] for l in lines])) if lines else 0.0,
        ocr_digit_ratio=sum(c.isdigit() for c in chars) / max(len(chars), 1),
        ocr_upper_ratio=sum(c.isupper() for c in chars) / max(len(chars), 1),
        ocr_text_area=area / float(H * W),
        ocr_median_height_mm=med_h / px_per_mm,
        skew_angle_abs=abs(skew),
    )
    feats.update(quality)
    feats["is_imported"] = float(rule.get("imported", False))
    for v in C.VIOLATIONS:
        feats[f"rv_{v}"] = float(v in rule["violations"])
    return feats


class LabelAnalyzer:
    def __init__(self, ocr_backend: str = "auto", threads: int | None = None):
        self.ocr = OCREngine(ocr_backend, threads=threads)

    def analyze(self, image, panel_width_mm: float = C.PANEL_WIDTH_MM, check_font: bool = True) -> dict:
        bgr = load_image(image)
        # phone photos are large: work at <= 1600 px wide (OCR geometry is mapped back below)
        pre = preprocess(bgr)
        segs = self.ocr.read(pre["ocr_input"])
        # bring geometry back to original-image pixels
        s = pre["scale"]
        for seg in segs:
            for k in ("height", "x0", "x1", "y0", "y1", "cy"):
                seg[k] /= s
            seg["box"] = (np.asarray(seg["box"]) / s).round(1).tolist()
        lines = group_lines(segs)
        out = features_from_lines(lines, bgr, panel_width_mm, pre["skew_angle"], check_font)
        out.update(segments=segs, preprocess=pre, ocr_backend=self.ocr.backend)
        return out

    def analyze_multi(self, images: list, panel_width_mm: float = C.PANEL_WIDTH_MM, check_font: bool = True) -> dict:
        """Analyse several photos of the SAME pack (front / back / side panels).
        Each photo is preprocessed and OCR'd separately, then the deskewed photos
        are stacked vertically and all OCR lines merged, so a declaration found on
        any panel counts."""
        if len(images) == 1:
            return self.analyze(images[0], panel_width_mm, check_font)
        import cv2

        parts, segs_all, skews, y_off, W0 = [], [], [], 0.0, None
        for im in images:
            bgr = load_image(im)
            pre = preprocess(bgr)
            segs = self.ocr.read(pre["ocr_input"])
            desk = pre["deskewed"]
            if W0 is None:
                W0 = desk.shape[1]
            r = W0 / desk.shape[1]  # bring every photo to a common width
            if r != 1:
                desk = cv2.resize(desk, (W0, int(round(desk.shape[0] * r))), interpolation=cv2.INTER_AREA)
            for sg in segs:
                for k in ("height", "x0", "x1"):
                    sg[k] *= r
                for k in ("y0", "y1", "cy"):
                    sg[k] = sg[k] * r + y_off
                b = np.asarray(sg["box"]) * r
                b[:, 1] += y_off
                sg["box"] = b.round(1).tolist()
            segs_all += segs
            parts.append(desk)
            skews.append(pre["skew_angle"])
            y_off += desk.shape[0]
        stacked = np.vstack(parts)
        lines = group_lines(segs_all)
        skew = float(np.mean(skews))
        out = features_from_lines(lines, stacked, panel_width_mm, skew, check_font)
        gray = cv2.cvtColor(stacked, cv2.COLOR_BGR2GRAY)
        out.update(segments=segs_all, ocr_backend=self.ocr.backend,
                   preprocess=dict(original=stacked, deskewed=stacked, enhanced=cv2.createCLAHE(2.0, (8, 8)).apply(gray),
                                   scale=1.0, skew_angle=skew))
        return out


def features_from_lines(lines: list[dict], bgr: np.ndarray, panel_width_mm: float, skew: float,
                        check_font: bool = True) -> dict:
    px_per_mm = bgr.shape[1] / float(panel_width_mm)
    rule = check_label(lines, px_per_mm, check_font=check_font)
    quality = image_quality_features(bgr)
    feats = build_features(lines, rule, quality, bgr.shape, px_per_mm, skew)
    return dict(features=feats, text=rule["full_text"], rule=rule, lines=lines, px_per_mm=px_per_mm)


def lines_from_cache(cached: list[dict]) -> list[dict]:
    """Rebuild OCR line dicts from the JSON cache written by extract_features."""
    out = []
    for l in cached:
        pts = np.array([p for b in l["boxes"] for p in b], dtype=float)
        out.append(dict(text=l["text"], conf=l["conf"], height=l["height"], boxes=l["boxes"],
                        x0=pts[:, 0].min(), x1=pts[:, 0].max(), y0=pts[:, 1].min(), y1=pts[:, 1].max()))
    return out
