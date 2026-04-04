"""
learn_router.py
Train a shallow decision tree router (with LR baseline comparison).

Input:  results/routing_dataset.csv
Output: results/router_model.joblib
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "routing_dataset.csv"
OUT_PATH = ROOT / "results" / "router_model.joblib"

FEATURE_COLS = [
    "H_final",
    "dH",
    "d2H",
    "dH_late",
    "jsd_early",
    "jsd_late",
    "jsd_conv",
    "mean_token_entropy",
    "max_spike_z",
]

METHOD_COST = {
    "greedy": 1.0,
    "iti": 1.5,
    "cove": 4.0,
}


def build_labels(df: pd.DataFrame) -> pd.Series:
    """
    Empirically-grounded labels based on routing_dataset.csv findings.
    General: greedy always wins.
    Medical + low d2H: CoVe wins.
    Medical + high d2H: ITI wins.
    Label distribution should be substantially less imbalanced than
    cheapest-correct labeling and therefore trainable.
    """
    d2h_threshold = df[df["domain"] == "medical"]["d2H"].median()

    labels = []
    for _, row in df.iterrows():
        if row["domain"] != "medical":
            labels.append("greedy")
        elif row["d2H"] <= d2h_threshold:
            labels.append("cove")
        else:
            labels.append("iti")

    return pd.Series(labels)


if __name__ == "__main__":
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing routing dataset: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH.name}")
    if "domain" in df:
        print(f"Domain distribution: {df['domain'].value_counts().to_dict()}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    x = df[FEATURE_COLS].values
    y = build_labels(df)

    print("\nLabel distribution:")
    print(y.value_counts())

    tree = DecisionTreeClassifier(
        max_depth=3,
        criterion="gini",
        min_samples_leaf=5,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tree_cv_acc = cross_val_score(tree, x, y, cv=cv, scoring="accuracy").mean()
    print(f"\nDecision Tree (max_depth=3) CV accuracy: {tree_cv_acc:.3f}")

    lr_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=0.1,
                    solver="lbfgs",
                    max_iter=500,
                    random_state=42,
                ),
            ),
        ]
    )
    lr_cv_acc = cross_val_score(lr_pipeline, x, y, cv=cv, scoring="accuracy").mean()
    print(f"Logistic Regression CV accuracy:        {lr_cv_acc:.3f}")

    if tree_cv_acc >= lr_cv_acc - 0.02:
        chosen = tree
        chosen_label = "DecisionTree"
    else:
        chosen = lr_pipeline
        chosen_label = "LogisticRegression"

    best_cv = max(tree_cv_acc, lr_cv_acc)
    print(f"\nChosen model: {chosen_label} (CV acc={best_cv:.3f})")

    chosen.fit(x, y)

    if chosen_label == "DecisionTree":
        print("\nLearned routing rules:")
        print(export_text(chosen, feature_names=FEATURE_COLS))

    print("\nFeature importance (tree) or coefficients (LR):")
    if chosen_label == "DecisionTree":
        for fname, imp in sorted(
            zip(FEATURE_COLS, chosen.feature_importances_),
            key=lambda pair: -pair[1],
        ):
            if imp > 0.01:
                print(f"  {fname:<22}: {imp:.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": chosen,
            "model_label": chosen_label,
            "feature_cols": FEATURE_COLS,
            "cv_acc": best_cv,
        },
        OUT_PATH,
    )
    print(f"\nSaved router to {OUT_PATH}")

    print("\nAccuracy by method and domain:")
    for domain in ["general", "medical"]:
        sub = df[df.get("domain", "") == domain] if "domain" in df else pd.DataFrame()
        if len(sub) == 0:
            continue
        print(f"  {domain} (n={len(sub)}):")
        print(f"    greedy: {(sub['greedy_ok'] == 1).mean():.1%}")
        print(f"    iti:    {(sub['iti_ok'] == 1).mean():.1%}")
        print(f"    cove:   {(sub['cove_ok'] == 1).mean():.1%}")
