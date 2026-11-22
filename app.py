"""Streamlit dashboard - Packaged Commodity Compliance Classification (SIH26034).

Run:  streamlit run app.py
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src import config as C

st.set_page_config(page_title="Label Compliance Checker - SIH26034", page_icon="📦", layout="wide")


@st.cache_resource(show_spinner="Loading OCR engine and models ...")
def get_predictor():
    from src.predict import CompliancePredictor

    return CompliancePredictor()


def models_ready() -> bool:
    return C.COMPLIANCE_MODEL_PATH.exists() and C.VIOLATION_MODEL_PATH.exists()


def draw_ocr(analysis: dict) -> np.ndarray:
    pre = analysis["preprocess"]
    img = pre["deskewed"]
    s = pre["scale"]
    if s != 1:
        img = cv2.resize(img, None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA)
    img = img.copy()
    F = analysis["rule"]["fields"]
    field_lines = {f["line"] for f in F.values() if f.get("found") and f.get("line") is not None}
    for i, ln in enumerate(analysis["lines"]):
        color = (40, 160, 40) if i in field_lines else (214, 120, 42)
        for b in ln["boxes"]:
            cv2.polylines(img, [np.array(b, dtype=np.int32)], True, color, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("📦 SIH26034")
st.sidebar.caption("Packaged Commodity Compliance Classification from Product Labels")
photo_mode = st.sidebar.radio(
    "Image type", ["📱 Real product photo", "🧪 Label with known size"],
    help="Real photos have no known physical scale, so the Rule 7 numeral-height check is skipped.") == "📱 Real product photo"
panel_mm = C.PANEL_WIDTH_MM
if not photo_mode:
    panel_mm = st.sidebar.number_input("Physical width of the image (mm)", 20.0, 500.0, C.PANEL_WIDTH_MM, 5.0,
                                       help="Converts font heights to millimetres for the Rule 7 check. "
                                            "Synthetic test-set labels set this automatically.")
threshold = st.sidebar.slider("Decision threshold P(non-compliant)", 0.1, 0.9, 0.5, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("**Team**\n\n- Dibyajit Choudhury (12516706)\n- Simran Meet Singh Bali (12518167)")

tab_an, tab_perf, tab_vio, tab_data, tab_rules = st.tabs(
    ["🔍 Scan / analyze", "📊 Model performance", "⚠️ Violation insights", "🗂️ Dataset", "📜 Rules"])

VERDICT_STYLE = {
    "COMPLIANT": ("✅", "#1baf7a", "All mandatory declarations were found and are valid."),
    "NON-COMPLIANT": ("❌", "#e34948", "One or more mandatory declarations are missing or invalid."),
    "NEEDS REVIEW": ("⚠️", "#eda100",
                     "The ML model and the rule engine disagree, so a human inspector should check this pack."),
}

# ------------------------------------------------------------------ analyze
with tab_an:
    st.header("Scan a packaged product")
    if not models_ready():
        st.error("Models not found. Run `python run_pipeline.py` first.")
    else:
        how = st.radio("Input", ["📁 Upload photos", "📷 Camera", "🧪 Test-set sample"], horizontal=True,
                       label_visibility="collapsed")
        images, sample = [], None
        if how == "📁 Upload photos":
            ups = st.file_uploader("Upload 1-4 photos of the SAME pack (front, back, sides). On a phone this "
                                   "also lets you take a photo with the back camera.",
                                   type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
            images = [u.getvalue() for u in (ups or [])][:4]
            if not images:
                demo = sorted(p for p in (C.ROOT / "samples").glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
                if demo and st.button(f"Try the demo pack ({len(demo)} photo(s) in samples/)"):
                    st.session_state["demo"] = True
                if st.session_state.get("demo"):
                    images = [str(p) for p in demo]
        elif how == "📷 Camera":
            st.caption("Hold the label flat, fill the frame, avoid glare. (Camera access needs HTTPS or localhost.)")
            shot = st.camera_input("Take a photo of the label")
            images = [shot.getvalue()] if shot else []
        else:
            pred_csv = C.REPORTS_DIR / "test_predictions.csv"
            if pred_csv.exists():
                tp = pd.read_csv(pred_csv)
                tp = tp[[(C.DATASET_DIR / f).exists() for f in tp.filename]]
                if tp.empty:
                    st.info("Test-set images are not included in this deployment. Run the pipeline locally to use them.")
                opts = ["-"] + [f"{r.image_id}  ({'compliant' if r.compliant else 'non-compliant'})" for r in tp.itertuples()]
                choice = st.selectbox("Pick an unseen test-set label", opts)
                if choice != "-":
                    sample = tp[tp.image_id == choice.split()[0]].iloc[0]
                    images = [str(C.DATASET_DIR / sample.filename)]

        if images:
            use_photo = photo_mode and sample is None
            pmm = panel_mm
            if sample is not None and C.LABELS_CSV.exists():
                lab = pd.read_csv(C.LABELS_CSV).set_index("image_id")
                if "panel_width_mm" in lab:
                    pmm = float(lab.loc[sample.image_id, "panel_width_mm"])
            with st.spinner("Preprocessing → OCR → feature extraction → classification ..."):
                res = get_predictor().predict(images if len(images) > 1 else images[0], panel_width_mm=pmm,
                                              threshold=threshold, check_font=not use_photo)
            a = res["analysis"]
            icon, color, msg = VERDICT_STYLE[res["verdict"]]
            msg = res.get("review_reason") or msg
            st.markdown(
                f"<div style='border-left:8px solid {color};background:{color}22;padding:14px 18px;border-radius:8px'>"
                f"<div style='font-size:1.9rem;font-weight:700'>{icon} {res['verdict']}</div>"
                f"<div style='opacity:.85'>{msg}</div></div>", unsafe_allow_html=True)
            st.write("")
            m1, m2, m3, m4 = st.columns(4)
            short = {"COMPLIANT": "✅ Pass", "NON-COMPLIANT": "❌ Fail", "NEEDS REVIEW": "⚠️ Unsure"}
            m1.metric("ML model", short[res["ml_verdict"]], help=f"{res['model']} (best of 5 trained classifiers)")
            m2.metric("P(non-compliant)", f"{res['prob_non_compliant']:.0%}")
            m3.metric("Rule engine", short[res["rule_engine"]["verdict"]])
            m4.metric("OCR lines read", len(a["lines"]))
            if sample is not None:
                truth = "COMPLIANT" if sample.compliant else "NON-COMPLIANT"
                vtxt = f" ({sample.violations.replace('|', ', ')})" if isinstance(sample.violations, str) else ""
                (st.success if truth == res["verdict"] else st.warning)(f"Ground truth: **{truth}**{vtxt}")

            if res["violations"]:
                st.subheader("Violations")
                for v in res["violations"]:
                    st.markdown(f"- **{v['title']}**: {v['detail']}  \n  <small>{v['rule']} · detected by "
                                f"{v['source']} · ML confidence {v['probability']:.0%}</small>", unsafe_allow_html=True)

            st.subheader("Mandatory declaration checklist")
            ck = pd.DataFrame(res["checklist"])
            ck.insert(0, "status", ck.pop("status").map({"ok": "✅", "missing": "❌", "unreadable": "⚠️ not readable"}))
            st.dataframe(ck, width="stretch", hide_index=True)
            if use_photo:
                st.caption("Rule 7 numeral-height check is skipped for phone photos (physical scale unknown). "
                           "Country of origin is only required for imported goods.")

            st.subheader("Image pipeline")
            i1, i2, i3 = st.columns(3)
            i1.image(cv2.cvtColor(a["preprocess"]["original"], cv2.COLOR_BGR2RGB), caption="Input", width="stretch")
            i2.image(a["preprocess"]["enhanced"], caption=f"Deskewed ({a['preprocess']['skew_angle']:+.1f}°) + CLAHE",
                     width="stretch")
            i3.image(draw_ocr(a), caption="OCR boxes (green = mandatory declaration)", width="stretch")

            with st.expander("Violation probabilities (multi-label ML model)"):
                vp = pd.Series(res["violation_probabilities"]).rename(index=C.VIOLATION_SHORT).sort_values()
                st.bar_chart(vp, horizontal=True)
            with st.expander("OCR text"):
                st.text(res["ocr_text"])
            with st.expander("Extracted features"):
                st.dataframe(pd.Series(a["features"], name="value"), width="stretch")
            report = {k: v for k, v in res.items() if k != "analysis"}
            st.download_button("⬇️ Download JSON report", json.dumps(report, indent=2, ensure_ascii=False),
                               file_name="compliance_report.json", mime="application/json")

# ------------------------------------------------------------------ performance
with tab_perf:
    st.header("Classification performance")
    if not C.METRICS_JSON.exists():
        st.info("Run the training pipeline to see metrics.")
    else:
        M = json.loads(C.METRICS_JSON.read_text())
        st.markdown(f"**Best model:** {M['best_model']} (selected by 5-fold cross-validated F1). "
                    f"Train = {M['meta']['n_train']} labels, held-out test = {M['meta']['n_test']} labels. "
                    "Positive class = *non-compliant*.")
        st.dataframe(pd.DataFrame(M["binary"]).T, width="stretch")
        for f in ["07_model_comparison.png", "08_confusion_matrices.png", "09_roc_curves.png", "10_feature_importance.png",
                  "13_sample_predictions.png"]:
            p = C.FIGURES_DIR / f
            if p.exists():
                st.image(str(p), width="stretch")
        st.text(M["classification_report"])

# ------------------------------------------------------------------ violations
with tab_vio:
    st.header("Common violations")
    if C.METRICS_JSON.exists():
        M = json.loads(C.METRICS_JSON.read_text())
        v = M["violations"]
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Micro-F1 (ML)", f"{v['ml']['micro_f1']:.3f}", f"{v['ml']['micro_f1'] - v['rules']['micro_f1']:+.3f} vs rules")
        a2.metric("Macro-F1 (ML)", f"{v['ml']['macro_f1']:.3f}")
        a3.metric("Hamming loss", f"{v['ml']['hamming_loss']:.3f}")
        a4.metric("Exact-match ratio", f"{v['ml']['exact_match']:.1%}")
        pv = pd.DataFrame(v["ml"]["per_violation"]).T.rename(index=C.VIOLATION_SHORT)
        st.dataframe(pv, width="stretch")
    for f in ["02_violation_frequency.png", "11_violation_detection_f1.png", "12_violation_actual_vs_predicted.png",
              "04_violation_cooccurrence.png", "05_violations_by_category.png", "06_netqty_font_check.png",
              "03_violations_per_label.png"]:
        p = C.FIGURES_DIR / f
        if p.exists():
            st.image(str(p), width="stretch")

# ------------------------------------------------------------------ dataset
with tab_data:
    st.header("Dataset")
    if C.FEATURES_CSV.exists():
        df = pd.read_csv(C.FEATURES_CSV)
        st.markdown(f"{len(df)} labelled label images · {df.compliant.mean():.0%} compliant · "
                    f"{df.category.nunique()} product categories")
        p = C.FIGURES_DIR / "01_dataset_overview.png"
        if p.exists():
            st.image(str(p), width="stretch")
        flt = st.multiselect("Filter by category", sorted(df.category.unique()))
        view = df[df.category.isin(flt)] if flt else df
        st.dataframe(view[["image_id", "category", "product", "brand", "compliant", "violations", "rule_violations",
                           "ocr_mean_conf", "deg_angle", "deg_blur", "deg_jpeg"]], width="stretch", hide_index=True)
        pick = st.selectbox("Preview label", view.image_id.head(200))
        r = view[view.image_id == pick].iloc[0]
        if (C.DATASET_DIR / r.filename).exists():
            st.image(str(C.DATASET_DIR / r.filename), width=420, caption=f"{r.image_id} — violations: {r.violations if isinstance(r.violations, str) else 'none'}")

# ------------------------------------------------------------------ rules
with tab_rules:
    st.header("Mandatory declarations checked")
    st.markdown("Based on the **Legal Metrology (Packaged Commodities) Rules, 2011** (simplified for this prototype).")
    st.dataframe(pd.DataFrame([dict(code=k, violation=v[0], reference=v[1], requirement=v[2])
                               for k, v in C.VIOLATION_INFO.items()]), width="stretch", hide_index=True)
    st.markdown("""
**Rule 7 – minimum height of net-quantity numerals**

| Net quantity | Minimum numeral height |
|---|---|
| up to 50 g / ml | 1 mm |
| 50 – 200 g / ml | 2 mm |
| 200 – 1000 g / ml | 4 mm |
| above 1 kg / L | 6 mm |
""")
