"""All evaluation and violation-analysis figures (saved to reports/figures)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve, auc  # noqa: E402

from src import config as C  # noqa: E402

# Validated categorical palette (fixed order) + neutral text colours
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COMPLIANT, NONCOMPLIANT = SERIES[0], SERIES[1]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "legend.frameon": False,
})


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)


def _vio_counts(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    return pd.Series({v: s.apply(lambda r: v in r.split("|")).sum() for v in C.VIOLATIONS})


# ---------------------------------------------------------------- dataset
def dataset_overview(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = df["compliant"].map({1: "Compliant", 0: "Non-compliant"}).value_counts().reindex(["Compliant", "Non-compliant"])
    bars = axes[0].bar(counts.index, counts.values, color=[COMPLIANT, NONCOMPLIANT], width=0.55)
    axes[0].bar_label(bars, labels=[f"{v} ({v / len(df):.0%})" for v in counts.values], color=INK, padding=3)
    axes[0].set_title("Class distribution")
    axes[0].set_ylabel("Labels")
    axes[0].grid(axis="x", visible=False)

    rate = df.groupby("category")["compliant"].agg(["mean", "size"]).sort_values("mean")
    y = np.arange(len(rate))
    axes[1].barh(y, rate["mean"], color=COMPLIANT, height=0.55, label="Compliant")
    axes[1].barh(y, 1 - rate["mean"], left=rate["mean"], color=NONCOMPLIANT, height=0.55, label="Non-compliant",
                 edgecolor="#fcfcfb", linewidth=2)
    axes[1].set_yticks(y, [f"{c} (n={n})" for c, n in zip(rate.index, rate["size"])])
    axes[1].set_xlim(0, 1)
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].set_title("Compliance rate by product category")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    axes[1].grid(axis="y", visible=False)
    _save(fig, "01_dataset_overview.png")


def violation_frequency(df: pd.DataFrame) -> None:
    nc = df[df["compliant"] == 0]
    counts = _vio_counts(nc["violations"]).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh([C.VIOLATION_SHORT[v] for v in counts.index], counts.values, color=NONCOMPLIANT, height=0.6)
    ax.bar_label(bars, labels=[f"{v}  ({v / len(nc):.0%} of non-compliant)" for v in counts.values],
                 padding=4, color=INK2, fontsize=9)
    ax.set_xlim(0, counts.max() * 1.45)
    ax.set_title(f"Most common violations  (n = {len(nc)} non-compliant labels)")
    ax.set_xlabel("Number of labels")
    ax.grid(axis="y", visible=False)
    _save(fig, "02_violation_frequency.png")

    per_label = nc["n_violations"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(per_label.index.astype(str), per_label.values, color=NONCOMPLIANT, width=0.5)
    ax.bar_label(bars, padding=3, color=INK2)
    ax.set_title("Violations per non-compliant label")
    ax.set_xlabel("Number of violations")
    ax.set_ylabel("Labels")
    ax.grid(axis="x", visible=False)
    _save(fig, "03_violations_per_label.png")


def violation_cooccurrence(df: pd.DataFrame) -> None:
    s = df["violations"].fillna("").astype(str)
    M = np.array([[int(v in r.split("|")) for v in C.VIOLATIONS] for r in s])
    co = M.T @ M
    names = [C.VIOLATION_SHORT[v] for v in C.VIOLATIONS]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(co, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            if co[i, j]:
                ax.text(j, i, co[i, j], ha="center", va="center", fontsize=8,
                        color="white" if co[i, j] > co.max() * 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Labels")
    ax.set_title("Violation co-occurrence (diagonal = total)")
    _save(fig, "04_violation_cooccurrence.png")


def violations_by_category(df: pd.DataFrame) -> None:
    rows = {}
    for cat, g in df.groupby("category"):
        rows[cat] = _vio_counts(g["violations"]) / len(g) * 100
    tab = pd.DataFrame(rows).T
    tab.columns = [C.VIOLATION_SHORT[v] for v in tab.columns]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(tab.values, cmap="Oranges", aspect="auto")
    ax.set_xticks(range(tab.shape[1]), tab.columns, rotation=35, ha="right")
    ax.set_yticks(range(tab.shape[0]), tab.index)
    ax.grid(False)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            ax.text(j, i, f"{tab.values[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if tab.values[i, j] > tab.values.max() * 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.03, label="% of labels in category")
    ax.set_title("Violation rate by product category")
    _save(fig, "05_violations_by_category.png")


def netqty_font_analysis(df: pd.DataFrame) -> None:
    d = df[(df["rule_net_qty"] == 1) & df["true_numeral_mm"].notna()].copy()
    d["too_small"] = d["violations"].fillna("").str.contains("NET_QTY_FONT_TOO_SMALL")
    fig, ax = plt.subplots(figsize=(7, 5))
    for flag, col, lab in [(False, COMPLIANT, "Numeral height OK"), (True, NONCOMPLIANT, "Too small (violation)")]:
        s = d[d["too_small"] == flag]
        ax.scatter(s["true_numeral_mm"] / s["required_numeral_mm"], s["netqty_height_ratio"], s=14, alpha=0.7,
                   color=col, label=lab, edgecolor="#fcfcfb", linewidth=0.5)
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.axvline(1, color=INK2, lw=1, ls="--")
    ax.set_xlabel("True numeral height / required (Rule 7)")
    ax.set_ylabel("Estimated from OCR box / required")
    ax.set_title("Net-quantity font-size check: OCR estimate vs ground truth")
    ax.legend(loc="upper left")
    _save(fig, "06_netqty_font_check.png")


# ---------------------------------------------------------------- models
def model_comparison(metrics: dict) -> None:
    res = metrics["binary"]
    models = list(res)
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    w = 0.8 / len(models)
    x = np.arange(len(keys))
    for i, m in enumerate(models):
        vals = [res[m].get(k, np.nan) for k in keys]
        vals = [0 if (v is None or v != v) else v for v in vals]
        b = ax.bar(x + i * w - 0.4 + w / 2, vals, w - 0.02, color=SERIES[i % len(SERIES)], label=m)
        if m == metrics["best_model"]:
            ax.bar_label(b, labels=[f"{v:.3f}" if v else "" for v in vals], fontsize=7, padding=2, color=INK)
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"])
    lo = min(v for m in models for k in keys if (v := res[m].get(k)) == v and v) - 0.05
    ax.set_ylim(max(0, lo), 1.02)
    ax.set_title(f"Model comparison on held-out test set  (best: {metrics['best_model']}, values labelled)")
    ax.legend(ncol=len(models), loc="lower center", bbox_to_anchor=(0.5, -0.25), fontsize=9)
    ax.grid(axis="x", visible=False)
    _save(fig, "07_model_comparison.png")


def confusion_matrices(pred: pd.DataFrame, metrics: dict) -> None:
    models = list(metrics["binary"])
    cols = 3
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
    for ax, m in zip(axes.flat, models):
        cm = confusion_matrix(pred["y_true"], pred[f"pred::{m}"], labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() * 0.55 else INK)
        ax.set_xticks([0, 1], ["Compliant", "Non-compl."])
        ax.set_yticks([0, 1], ["Compliant", "Non-compl."])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        star = "  ★ best" if m == metrics["best_model"] else ""
        ax.set_title(f"{m}{star}\nF1 = {metrics['binary'][m]['f1']:.3f}", fontsize=10)
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")
    _save(fig, "08_confusion_matrices.png")


def roc_curves(pred: pd.DataFrame, metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, m in enumerate(metrics["binary"]):
        col = f"proba::{m}"
        if col not in pred:
            continue
        fpr, tpr, _ = roc_curve(pred["y_true"], pred[col])
        ax.plot(fpr, tpr, lw=2, color=SERIES[i % len(SERIES)], label=f"{m} (AUC {auc(fpr, tpr):.3f})")
    rb = metrics["binary"].get("Rule-based baseline")
    if rb:
        tn_fp = pred[pred.y_true == 0]
        fpr_r = (tn_fp["pred::Rule-based baseline"] == 1).mean()
        tpr_r = rb["recall"]
        ax.scatter([fpr_r], [tpr_r], s=60, color=INK, zorder=5, label="Rule-based baseline (single point)")
    ax.plot([0, 1], [0, 1], ls="--", color=INK2, lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (non-compliant detected)")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right", fontsize=8.5)
    _save(fig, "09_roc_curves.png")


def feature_importance(top: int = 20) -> None:
    imp = pd.read_csv(C.REPORTS_DIR / "feature_importance.csv", index_col=0).iloc[:, 0].head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(imp.index, imp.values, color=COMPLIANT, height=0.6)
    ax.set_title(f"Top {top} engineered features (Random Forest importance)")
    ax.set_xlabel("Mean decrease in impurity")
    ax.grid(axis="y", visible=False)
    _save(fig, "10_feature_importance.png")


def violation_detection(metrics: dict) -> None:
    ml, rules = metrics["violations"]["ml"]["per_violation"], metrics["violations"]["rules"]["per_violation"]
    names = [C.VIOLATION_SHORT[v] for v in C.VIOLATIONS]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y + 0.2, [ml[v]["f1"] for v in C.VIOLATIONS], 0.38, color=SERIES[0], label="ML (Gradient Boosting + rule features)")
    ax.barh(y - 0.2, [rules[v]["f1"] for v in C.VIOLATIONS], 0.38, color=SERIES[1], label="Rule engine")
    ax.set_yticks(y, [f"{n} (n={ml[v]['support']})" for n, v in zip(names, C.VIOLATIONS)])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-score on test set")
    ax.set_title(f"Per-violation detection  (micro-F1: ML {metrics['violations']['ml']['micro_f1']:.3f}, "
                 f"rules {metrics['violations']['rules']['micro_f1']:.3f})")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.grid(axis="y", visible=False)
    _save(fig, "11_violation_detection_f1.png")


def predicted_vs_actual(pred: pd.DataFrame) -> None:
    actual = _vio_counts(pred["violations"])
    predicted = pd.Series({v: int(pred[f"vio_pred::{v}"].sum()) for v in C.VIOLATIONS})
    order = actual.sort_values().index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y + 0.2, actual[order], 0.38, color=SERIES[0], label="Actual")
    ax.barh(y - 0.2, predicted[order], 0.38, color=SERIES[2], label="Predicted by model")
    ax.set_yticks(y, [C.VIOLATION_SHORT[v] for v in order])
    ax.set_xlabel("Labels in test set")
    ax.set_title("Violation frequency on test set: actual vs predicted")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, "12_violation_actual_vs_predicted.png")


def sample_predictions(pred: pd.DataFrame, metrics: dict, n: int = 6) -> None:
    best = metrics["best_model"]
    rng = np.random.default_rng(C.RANDOM_SEED)
    picks = pd.concat([pred[pred.y_true == 0].sample(n // 2, random_state=1),
                       pred[pred.y_true == 1].sample(n - n // 2, random_state=2)])
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5.2 * ((n + 1) // 2), 13))
    for ax, (_, r) in zip(axes.flat, picks.iterrows()):
        img = Image.open(C.DATASET_DIR / r.filename).convert("RGB")
        ax.imshow(img)
        ocr_file = C.OCR_CACHE_DIR / f"{r.image_id}.json"
        if ocr_file.exists():
            for ln in json.loads(ocr_file.read_text(encoding="utf-8")):
                for b in ln["boxes"]:
                    b = np.array(b + [b[0]])
                    ax.plot(b[:, 0], b[:, 1], color=SERIES[2], lw=0.8)
        p_nc = r[f"proba::{best}"]
        verdict = "NON-COMPLIANT" if p_nc >= 0.5 else "COMPLIANT"
        truth = "NON-COMPLIANT" if r.y_true else "COMPLIANT"
        vio = [C.VIOLATION_SHORT[v] for v in C.VIOLATIONS if r[f"vio_pred::{v}"] == 1]
        ok = "correct" if (p_nc >= 0.5) == bool(r.y_true) else "WRONG"
        ax.set_title(f"{r.image_id}  [{ok}]\nPredicted: {verdict} (p={p_nc:.2f}) | Actual: {truth}\n"
                     + ("; ".join(vio) if vio else "no violations predicted"), fontsize=9)
        ax.axis("off")
        _ = rng
    _save(fig, "13_sample_predictions.png")


def make_all() -> None:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.FEATURES_CSV)
    metrics = json.loads(C.METRICS_JSON.read_text())
    pred = pd.read_csv(C.REPORTS_DIR / "test_predictions.csv")
    dataset_overview(df)
    violation_frequency(df)
    violation_cooccurrence(df)
    violations_by_category(df)
    netqty_font_analysis(df)
    model_comparison(metrics)
    confusion_matrices(pred, metrics)
    roc_curves(pred, metrics)
    feature_importance()
    violation_detection(metrics)
    predicted_vs_actual(pred)
    sample_predictions(pred, metrics)
    print(f"  figures written to {C.FIGURES_DIR}")


if __name__ == "__main__":
    make_all()
