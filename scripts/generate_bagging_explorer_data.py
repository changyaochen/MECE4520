"""Generate a compact, precomputed payload for the bagging explorer."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


FEATURES = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
TARGET = "high_co"
RANDOM_STATE = 4520
N_TREES = 200
MAX_DEPTH = 5
# This position is fixed from the reproducible test split before fitting any
# bagged model. It is not chosen from its fitted probability or classification.
FIXED_TEST_POSITION = 4520
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "site" / "data" / "gas-turbine-course.csv"
OUTPUT_PATH = ROOT / "site" / "assets" / "bagging-explorer.js"


def rounded(values: Sequence[float] | np.ndarray, digits: int = 5) -> list[float]:
    """Convert numeric values into compact JSON-ready floats.

    Args:
        values: Values to serialize.
        digits: Number of decimal places to retain.

    Returns:
        Rounded Python floats.
    """
    return [round(float(value), digits) for value in values]


def cumulative_auc(*, tree_probability: np.ndarray, labels: pd.Series) -> np.ndarray:
    """Calculate held-out AUC as successive tree predictions are averaged.

    Args:
        tree_probability: One probability column for each fitted tree.
        labels: Held-out binary labels.

    Returns:
        AUC values for ensembles containing one through all trees.
    """
    tree_counts = np.arange(1, tree_probability.shape[1] + 1)
    running_probability = np.cumsum(tree_probability, axis=1) / tree_counts
    return np.array(
        [
            roc_auc_score(labels, running_probability[:, index])
            for index in range(tree_probability.shape[1])
        ]
    )


def bootstrap_summary(*, sample_positions: np.ndarray, training_size: int) -> dict[str, object]:
    """Summarize one actual bootstrap sample without serializing all draws.

    Args:
        sample_positions: Training-set positions drawn with replacement.
        training_size: Number of rows in the original training dataset.

    Returns:
        Counts describing one bootstrap dataset.
    """
    counts = np.bincount(sample_positions, minlength=training_size)
    frequency = {
        "0": int(np.sum(counts == 0)),
        "1": int(np.sum(counts == 1)),
        "2": int(np.sum(counts == 2)),
        "3": int(np.sum(counts == 3)),
        "4+": int(np.sum(counts >= 4)),
    }
    return {
        "draws": int(len(sample_positions)),
        "uniqueRows": int(np.sum(counts > 0)),
        "omittedRows": int(np.sum(counts == 0)),
        "frequency": frequency,
    }


def root_details(*, tree: DecisionTreeClassifier, positive_fraction: float) -> dict[str, object]:
    """Extract a concise description of a fitted tree's root node.

    Args:
        tree: A fitted decision tree.
        positive_fraction: Positive-label fraction in its bootstrap training set.

    Returns:
        Root feature, cutpoint, impurity, and positive-label fraction.
    """
    root_feature_index = int(tree.tree_.feature[0])
    return {
        "feature": FEATURES[root_feature_index],
        "threshold": round(float(tree.tree_.threshold[0]), 3),
        "gini": round(float(tree.tree_.impurity[0]), 3),
        "positiveFraction": round(float(positive_fraction), 3),
    }


def feature_importances(*, tree: DecisionTreeClassifier) -> dict[str, float]:
    """Return the relative total Gini reduction attributed to each feature.

    Args:
        tree: A fitted decision tree.

    Returns:
        One normalized feature-importance value for each course feature.
    """
    return {
        feature: round(float(importance), 5)
        for feature, importance in zip(FEATURES, tree.feature_importances_)
    }


def main() -> None:
    """Fit real bagged trees and write the data used by the web explorer."""
    data = pd.read_csv(DATA_PATH)
    model_data = data[FEATURES + [TARGET]].dropna().copy()
    X_train, X_test, y_train, y_test = train_test_split(
        model_data[FEATURES],
        model_data[TARGET],
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=model_data[TARGET],
    )

    single_tree = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    single_tree.fit(X=X_train, y=y_train)
    single_tree_auc = roc_auc_score(y_test, single_tree.predict_proba(X_test)[:, 1])

    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH),
        n_estimators=N_TREES,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    bagging.fit(X=X_train, y=y_train)

    X_test_values = X_test.to_numpy()
    tree_probability = np.column_stack(
        [tree.predict_proba(X_test_values)[:, 1] for tree in bagging.estimators_]
    )
    y_train_values = y_train.to_numpy()

    tree_summaries = []
    for tree, sample_positions, probability in zip(
        bagging.estimators_, bagging.estimators_samples_, tree_probability.T
    ):
        tree_summaries.append(
            {
                "bootstrap": bootstrap_summary(
                    sample_positions=sample_positions,
                    training_size=len(X_train),
                ),
                "root": root_details(
                    tree=tree,
                    positive_fraction=y_train_values[sample_positions].mean(),
                ),
                "featureImportances": feature_importances(tree=tree),
                "testAuc": round(float(roc_auc_score(y_test, probability)), 5),
            }
        )

    fixed_row = X_test.iloc[FIXED_TEST_POSITION]
    prediction_observation = {
        "actualLabel": int(y_test.iloc[FIXED_TEST_POSITION]),
        "features": {
            feature: round(float(fixed_row[feature]), 2)
            for feature in FEATURES
        },
        "treeProbabilities": rounded(tree_probability[FIXED_TEST_POSITION]),
    }

    payload = {
        "metadata": {
            "nTrees": N_TREES,
            "nFeatures": len(FEATURES),
            "nTraining": int(len(X_train)),
            "nTest": int(len(X_test)),
        },
        "singleTree": {"testAuc": round(float(single_tree_auc), 5)},
        "ensemble": {
            "cumulativeTestAuc": rounded(
                cumulative_auc(tree_probability=tree_probability, labels=y_test)
            ),
            "trees": tree_summaries,
        },
        "predictionObservation": prediction_observation,
    }

    OUTPUT_PATH.write_text(
        "window.MECE4520BaggingExplorer = "
        + json.dumps(payload, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
