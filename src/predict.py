"""Inference: classify a product-label image as compliant / non-compliant and
list the violations, combining the trained ML models with the explainable
rule engine.

Usage:  python -m src.predict path/to/label.jpg [--panel-mm 100] [--json]
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.features import NUMERIC_FEATURES, TEXT_FEATURE, LabelAnalyzer


class CompliancePredictor:
    def __init__(self, analyzer: LabelAnalyzer | None = None):
        self.analyzer = analyzer or LabelAnalyzer()
        self.bin = joblib.load(C.COMPLIANCE_MODEL_PATH)
        self.vio = joblib.load(C.VIOLATION_MODEL_PATH)

    def predict(self, image, panel_width_mm: float = C.PANEL_WIDTH_MM, threshold: float = 0.5,
                check_font: bool = True) -> dict:
        """`image` may be a single image or a list of photos of the same pack."""
        images = image if isinstance(image, (list, tuple)) else [image]
        a = self.analyzer.analyze_multi(list(images), panel_width_mm=panel_width_mm, check_font=check_font)
        row = pd.DataFrame([{**{k: a["features"][k] for k in NUMERIC_FEATURES}, TEXT_FEATURE: a["text"].replace("\n", " | ")}])
        p_nc = float(self.bin["model"].predict_proba(row)[0, 1])

        probas = self.vio["model"].predict_proba(row)
        classes = self.vio["model"].named_steps["clf"].classes_
        vio_p = {}
        for v, pr, cls in zip(self.vio["violations"], probas, classes):
            vio_p[v] = float(pr[0, list(cls).index(1)]) if 1 in cls else 0.0
        if not check_font:  # physical scale unknown -> font-size violation cannot be assessed
            vio_p["NET_QTY_FONT_TOO_SMALL"] = 0.0
        ml_vios = [v for v, p in sorted(vio_p.items(), key=lambda kv: -kv[1]) if p >= threshold]
        ml_nc = p_nc >= threshold
        rule_vios = a["rule"]["violations"]
        rule_nc = bool(rule_vios)

        # Hybrid decision: ML model and explainable rule engine must agree,
        # otherwise the label is flagged for a human inspector.
        unreadable = a["rule"].get("unreadable", [])
        # An unreadable value can't be judged either way: drop ML guesses about it.
        unjudgeable = {"net_qty": {"MISSING_NET_QUANTITY", "NON_STANDARD_UNIT", "NET_QTY_FONT_TOO_SMALL"},
                       "mrp": {"MISSING_MRP"}, "mfg_date": {"MISSING_MFG_DATE"}}
        skip = set().union(*(unjudgeable.get(u, set()) for u in unreadable)) if unreadable else set()
        ml_vios = [v for v in ml_vios if v not in skip]
        names = {"net_qty": "net quantity", "mrp": "MRP value", "mfg_date": "manufacturing date"}
        if ml_nc and rule_nc:
            final, reason = "NON-COMPLIANT", ""
        elif not ml_nc and not rule_nc and not unreadable:
            final, reason = "COMPLIANT", ""
        else:
            final = "NEEDS REVIEW"
            if unreadable:
                reason = ("Some printed values could not be read (cut off, curved or blurred): "
                          + ", ".join(names.get(u, u) for u in unreadable)
                          + ". Retake the photo with that part of the pack flat, in focus and fully in the frame.")
            else:
                reason = "The ML model and the rule engine disagree, so a human inspector should check this pack."
        codes = list(dict.fromkeys(rule_vios + ml_vios))
        source = {v: ("ML + rules" if v in rule_vios and v in ml_vios else "rules" if v in rule_vios else "ML")
                  for v in codes}

        F = a["rule"]["fields"]
        nq = F["net_qty"]
        checks = [
            ("Manufacturer / packer name & address", F["manufacturer"]),
            ("Net quantity", nq),
            ("Retail sale price (MRP)", F["mrp"]),
            ("MRP inclusive of all taxes", F["mrp_tax"]),
            ("Month & year of manufacture / packing", F["mfg_date"]),
            ("Best before / use by", F["best_before"]),
            ("Consumer care details", F["consumer_care"]),
        ]
        if a["rule"]["imported"] or F["origin"]["found"]:
            checks.append(("Country of origin (imported goods)", F["origin"]))
        key_of = {id(F[k]): k for k in F}
        def status(f):
            if key_of.get(id(f)) in unreadable:
                return "unreadable"
            return "ok" if f["found"] else "missing"
        checklist = [dict(declaration=n, status=status(f), evidence=f.get("value", "")[:120],
                          keyword_similarity=round(f["fuzzy"], 2)) for n, f in checks]
        if nq["found"]:
            q = nq["qty"]
            checklist.append(dict(declaration="Standard SI unit", status="ok" if q["standard"] else "missing",
                                  evidence=f"{q['value']:g} {q['unit']}", keyword_similarity=None))
            if check_font:
                checklist.append(dict(declaration="Numeral height (Rule 7)",
                                      status="ok" if nq["numeral_mm"] >= nq["required_mm"] * 0.97 else "missing",
                                      evidence=f"~{nq['numeral_mm']:.1f} mm (required >= {nq['required_mm']:.0f} mm)",
                                      keyword_similarity=None))

        return dict(
            verdict=final,
            review_reason=reason,
            unreadable=unreadable,
            ml_verdict="NON-COMPLIANT" if ml_nc else "COMPLIANT",
            prob_non_compliant=round(p_nc, 4),
            model=self.bin["name"],
            violations=[dict(code=v, title=C.VIOLATION_INFO[v][0], rule=C.VIOLATION_INFO[v][1],
                             detail=C.VIOLATION_INFO[v][2], probability=round(vio_p[v], 3), source=source[v])
                        for v in codes],
            violation_probabilities={k: round(v, 3) for k, v in vio_p.items()},
            rule_engine=dict(verdict="NON-COMPLIANT" if rule_nc else ("NEEDS REVIEW" if unreadable else "COMPLIANT"),
                             violations=a["rule"]["violations"]),
            checklist=checklist,
            ocr_text=a["text"],
            ocr_backend=a["ocr_backend"],
            analysis=a,
        )


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles: allow the rupee sign
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="+", help="one or more photos of the same pack")
    ap.add_argument("--panel-mm", type=float, default=C.PANEL_WIDTH_MM, help="physical width of the label in mm")
    ap.add_argument("--json", action="store_true", help="print the full JSON report")
    ap.add_argument("--photo", action="store_true", help="real photo: skip the font-height check (scale unknown)")
    a = ap.parse_args()
    res = CompliancePredictor().predict(a.image, panel_width_mm=a.panel_mm, check_font=not a.photo)
    res.pop("analysis")
    if a.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    print(f"\nFINAL VERDICT: {res['verdict']}")
    print(f"  ML model ({res['model']}): {res['ml_verdict']}  (P(non-compliant) = {res['prob_non_compliant']:.2f})")
    print(f"  Rule engine: {res['rule_engine']['verdict']}")
    if res["review_reason"]:
        print(f"  Note: {res['review_reason']}")
    print("\nMandatory declarations:")
    for c in res["checklist"]:
        tag = {"ok": "OK", "missing": "XX", "unreadable": "??"}[c["status"]]
        print(f"  [{tag}] {c['declaration']:<40} {c['evidence']}")
    if res["violations"]:
        print("\nViolations:")
        for v in res["violations"]:
            print(f"  - {v['title']}  ({v['rule']}; found by {v['source']}, ML p={v['probability']:.2f})")


if __name__ == "__main__":
    main()
