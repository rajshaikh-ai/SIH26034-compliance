"""Train and evaluate compliance classifiers.

Task A - binary: is the label NON-COMPLIANT (positive class) or compliant?
    Models: Logistic Regression, SVM (RBF), Random Forest, Gradient Boosting,
    MLP neural network - compared against the explainable rule-based baseline.
Task B - multi-label: which of the 10 violations are present?
    Model: Random Forest (multi-output) vs. rule engine.

Features = hand-crafted rule/OCR/image features (scaled) + char n-gram TF-IDF
of the OCR text compressed with truncated SVD (LSA).

Usage:  python -m src.train
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             hamming_loss, precision_recall_fscore_support, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config as C
from src.features import MODEL_FEATURES, NUMERIC_FEATURES, TEXT_FEATURE


def make_preprocessor() -> ColumnTransformer:
    """Scaled engineered features only. (A char n-gram TF-IDF block of the raw
    OCR text was tried: it learns the synthetic labels' vocabulary - brand and
    product names - and made models over-confident on real product photos.)"""
    return ColumnTransformer([("num", StandardScaler(), MODEL_FEATURES)])


# Simplest first: used to break near-ties in model selection, because simpler
# models extrapolate more gracefully to real photos than the training data.
SIMPLICITY_ORDER = ["Logistic Regression", "SVM (RBF)", "Random Forest", "Gradient Boosting", "MLP Neural Net"]
SELECTION_TOLERANCE = 0.01


def candidate_models() -> dict:
    s = C.RANDOM_SEED
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"),
        "SVM (RBF)": CalibratedClassifierCV(SVC(C=3.0, gamma="scale", class_weight="balanced", random_state=s),
                                            method="sigmoid", ensemble=False),
        "Random Forest": RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=s),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                            class_weight="balanced", random_state=s),
        "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=800,
                                        early_stopping=True, random_state=s),
    }


def violation_matrix(series: pd.Series) -> np.ndarray:
    rows = series.fillna("").astype(str)
    return np.array([[int(v in r.split("|")) for v in C.VIOLATIONS] for r in rows])


def binary_metrics(y, pred, proba=None) -> dict:
    m = dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
             recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0))
    m["roc_auc"] = roc_auc_score(y, proba) if proba is not None else float("nan")
    return {k: round(float(v), 4) for k, v in m.items()}


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(C.FEATURES_CSV)
    df[TEXT_FEATURE] = df[TEXT_FEATURE].fillna("")
    df["violations"] = df["violations"].fillna("")
    df["rule_violations"] = df["rule_violations"].fillna("")
    df["non_compliant"] = 1 - df["compliant"].astype(int)
    return df


def main() -> None:
    C.ensure_dirs()
    df = load_dataset()
    X = df[NUMERIC_FEATURES + [TEXT_FEATURE]]
    y = df["non_compliant"].values
    Y = violation_matrix(df["violations"])
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=y, random_state=C.RANDOM_SEED)
    Xtr, Xte, ytr, yte = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]
    print(f"Dataset: {len(df)} labels | train {len(idx_tr)} | test {len(idx_te)} | "
          f"non-compliant {y.mean():.1%}")

    # ------------------------------------------------------------ Task A
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_SEED)
    results, fitted, test_outputs = {}, {}, {}
    for name, clf in candidate_models().items():
        t0 = time.time()
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
        cv_f1 = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1", n_jobs=1)
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = binary_metrics(yte, pred, proba)
        m.update(cv_f1_mean=round(float(cv_f1.mean()), 4), cv_f1_std=round(float(cv_f1.std()), 4),
                 train_seconds=round(time.time() - t0, 1))
        results[name], fitted[name] = m, pipe
        test_outputs[name] = dict(pred=pred.tolist(), proba=proba.round(5).tolist())
        print(f"  {name:<20} F1={m['f1']:.3f}  AUC={m['roc_auc']:.3f}  CV-F1={m['cv_f1_mean']:.3f}±{m['cv_f1_std']:.3f}")

    rule_pred = (1 - df["rule_compliant"].values[idx_te]).astype(int)
    results["Rule-based baseline"] = binary_metrics(yte, rule_pred)
    test_outputs["Rule-based baseline"] = dict(pred=rule_pred.tolist(), proba=None)
    print(f"  {'Rule-based baseline':<20} F1={results['Rule-based baseline']['f1']:.3f}")

    top = max(results[n]["cv_f1_mean"] for n in fitted)
    best = next(n for n in SIMPLICITY_ORDER if n in fitted and results[n]["cv_f1_mean"] >= top - SELECTION_TOLERANCE)
    print(f"Selected model: {best} (simplest within {SELECTION_TOLERANCE} of the best 5-fold CV F1 {top:.3f})")
    best_pred = np.array(test_outputs[best]["pred"])
    report_txt = classification_report(yte, best_pred, target_names=["Compliant", "Non-compliant"], digits=4)

    # ------------------------------------------------------------ Task B
    Y_rule = violation_matrix(df["rule_violations"].iloc[idx_te])
    Yte = Y[idx_te]
    vio_candidates = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, class_weight="balanced",
                                                            random_state=C.RANDOM_SEED),
    }
    vio_scores, vio_fitted = {}, {}
    for name, est in vio_candidates.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", MultiOutputClassifier(est, n_jobs=-1))])
        # select on a validation split of the TRAINING data, never on the test set
        a, b = train_test_split(np.arange(len(idx_tr)), test_size=0.2, random_state=C.RANDOM_SEED)
        pipe.fit(Xtr.iloc[a], Y[idx_tr][a])
        vio_scores[name] = f1_score(Y[idx_tr][b], pipe.predict(Xtr.iloc[b]), average="micro")
        vio_fitted[name] = pipe
    vtop = max(vio_scores.values())
    vio_name = next(n for n in vio_candidates if vio_scores[n] >= vtop - SELECTION_TOLERANCE)
    print("Violation model validation micro-F1:", {k: round(v, 3) for k, v in vio_scores.items()}, "->", vio_name)
    vio_pipe = Pipeline([("prep", make_preprocessor()),
                         ("clf", MultiOutputClassifier(vio_candidates[vio_name], n_jobs=-1))])
    vio_pipe.fit(Xtr, Y[idx_tr])
    Yp = vio_pipe.predict(Xte)

    def per_violation(Ypred):
        p, r, f, s = precision_recall_fscore_support(Yte, Ypred, zero_division=0)
        return {v: dict(precision=round(float(p[i]), 4), recall=round(float(r[i]), 4), f1=round(float(f[i]), 4),
                        support=int(s[i])) for i, v in enumerate(C.VIOLATIONS)}

    vio_results = {
        "ml": dict(per_violation=per_violation(Yp), micro_f1=round(f1_score(Yte, Yp, average="micro"), 4),
                   macro_f1=round(f1_score(Yte, Yp, average="macro", zero_division=0), 4),
                   hamming_loss=round(hamming_loss(Yte, Yp), 4),
                   exact_match=round(float((Yp == Yte).all(axis=1).mean()), 4)),
        "rules": dict(per_violation=per_violation(Y_rule), micro_f1=round(f1_score(Yte, Y_rule, average="micro"), 4),
                      macro_f1=round(f1_score(Yte, Y_rule, average="macro", zero_division=0), 4),
                      hamming_loss=round(hamming_loss(Yte, Y_rule), 4),
                      exact_match=round(float((Y_rule == Yte).all(axis=1).mean()), 4)),
    }
    vio_results["ml"]["model"] = vio_name
    print(f"Violation model micro-F1={vio_results['ml']['micro_f1']:.3f} "
          f"(rules {vio_results['rules']['micro_f1']:.3f})")

    # ------------------------------------------------------------ Save
    meta = dict(trained_at=datetime.now().isoformat(timespec="seconds"), n_train=len(idx_tr), n_test=len(idx_te),
                numeric_features=NUMERIC_FEATURES, text_feature=TEXT_FEATURE, positive_class="non_compliant")
    joblib.dump(dict(model=fitted[best], name=best, metrics=results[best], **meta), C.COMPLIANCE_MODEL_PATH)
    joblib.dump(dict(model=vio_pipe, violations=C.VIOLATIONS, metrics=vio_results["ml"], **meta), C.VIOLATION_MODEL_PATH)
    # refit a copy of every model is unnecessary; keep only the best for inference

    metrics = dict(meta=meta, best_model=best, binary=results, violations=vio_results,
                   confusion_matrix=confusion_matrix(yte, best_pred).tolist(), classification_report=report_txt)
    C.METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    (C.REPORTS_DIR / "classification_report.txt").write_text(f"Best model: {best}\n\n{report_txt}")
    pd.DataFrame(results).T.to_csv(C.REPORTS_DIR / "model_comparison.csv")

    test_df = df.iloc[idx_te][["image_id", "filename", "category", "compliant", "violations", "rule_violations"]].copy()
    test_df["y_true"] = yte
    for name, out in test_outputs.items():
        test_df[f"pred::{name}"] = out["pred"]
        if out["proba"] is not None:
            test_df[f"proba::{name}"] = out["proba"]
    for j, v in enumerate(C.VIOLATIONS):
        test_df[f"vio_pred::{v}"] = Yp[:, j]
    test_df.to_csv(C.REPORTS_DIR / "test_predictions.csv", index=False)

    # Permutation-free importance: RF impurity importances on the numeric block
    rf = fitted["Random Forest"]
    imp = rf.named_steps["clf"].feature_importances_[: len(MODEL_FEATURES)]
    pd.Series(imp, index=MODEL_FEATURES).sort_values(ascending=False).to_csv(C.REPORTS_DIR / "feature_importance.csv")

    from src import visualize
    visualize.make_all()
    print(f"Models saved to {C.MODELS_DIR}\nReports & figures saved to {C.REPORTS_DIR}")


if __name__ == "__main__":
    main()
