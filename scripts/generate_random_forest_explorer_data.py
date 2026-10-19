"""Generate the compact, precomputed data payload for the RF explorer."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


FEATURES = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
TARGET = "high_co"
RANDOM_STATE = 4520
N_TREES = 200
MAX_DEPTH = 5
FEATURE_CANDIDATES_PER_SPLIT = 3
# This position is fixed from the reproducible test split before fitting any
# random forest. It is not selected from a fitted probability or prediction.
FIXED_TEST_POSITION = 4520
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "site" / "data" / "gas-turbine-course.csv"
OUTPUT_PATH = ROOT / "site" / "assets" / "random-forest-explorer.js"


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
        AUC values for forests containing one through all trees.
    """
    tree_counts = np.arange(1, tree_probability.shape[1] + 1)
    running_probability = np.cumsum(tree_probability, axis=1) / tree_counts
    return np.array(
        [
            roc_auc_score(labels, running_probability[:, index])
            for index in range(tree_probability.shape[1])
        ]
    )


def bootstrap_summary(
    *, sample_positions: np.ndarray, training_size: int
) -> dict[str, int]:
    """Summarize one actual bootstrap sample without serializing all draws.

    Args:
        sample_positions: Training-set positions drawn with replacement.
        training_size: Number of rows in the original training dataset.

    Returns:
        Counts describing one bootstrap dataset.
    """
    counts = np.bincount(sample_positions, minlength=training_size)
    return {
        "draws": int(len(sample_positions)),
        "uniqueRows": int(np.sum(counts > 0)),
        "omittedRows": int(np.sum(counts == 0)),
    }


def root_details(*, tree: DecisionTreeClassifier) -> dict[str, object]:
    """Extract a concise description of a fitted tree's root node.

    Args:
        tree: A fitted decision tree.

    Returns:
        Root feature, cutpoint, and impurity.
    """
    root_feature_index = int(tree.tree_.feature[0])
    return {
        "feature": FEATURES[root_feature_index],
        "threshold": round(float(tree.tree_.threshold[0]), 3),
        "gini": round(float(tree.tree_.impurity[0]), 3),
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


def unused_features(*, tree: DecisionTreeClassifier) -> list[str]:
    """Identify features that do not appear in any split of one tree.

    Args:
        tree: A fitted decision tree.

    Returns:
        Feature names never selected at any internal node.
    """
    used_indices = set(tree.tree_.feature[tree.tree_.feature >= 0])
    return [
        feature for index, feature in enumerate(FEATURES) if index not in used_indices
    ]


def main() -> None:
    """Fit a random forest and write the data used by the web explorer."""
    data = pd.read_csv(DATA_PATH)
    model_data = data[FEATURES + [TARGET]].dropna().copy()
    X_train, X_test, y_train, y_test = train_test_split(
        model_data[FEATURES],
        model_data[TARGET],
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=model_data[TARGET],
    )

    forest = RandomForestClassifier(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        max_features=FEATURE_CANDIDATES_PER_SPLIT,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    forest.fit(X=X_train, y=y_train)

    X_test_values = X_test.to_numpy()
    tree_probability = np.column_stack(
        [tree.predict_proba(X_test_values)[:, 1] for tree in forest.estimators_]
    )

    tree_summaries = []
    for tree, sample_positions, probability in zip(
        forest.estimators_, forest.estimators_samples_, tree_probability.T
    ):
        tree_summaries.append(
            {
                "bootstrap": bootstrap_summary(
                    sample_positions=sample_positions,
                    training_size=len(X_train),
                ),
                "root": root_details(tree=tree),
                "featureImportances": feature_importances(tree=tree),
                "unusedFeatures": unused_features(tree=tree),
                "testAuc": round(float(roc_auc_score(y_test, probability)), 5),
            }
        )

    fixed_row = X_test.iloc[FIXED_TEST_POSITION]
    prediction_observation = {
        "actualLabel": int(y_test.iloc[FIXED_TEST_POSITION]),
        "features": {
            feature: round(float(fixed_row[feature]), 2) for feature in FEATURES
        },
        "treeProbabilities": rounded(tree_probability[FIXED_TEST_POSITION]),
    }

    payload = {
        "metadata": {
            "nTrees": N_TREES,
            "nFeatures": len(FEATURES),
            "nTraining": int(len(X_train)),
            "nTest": int(len(X_test)),
            "maxDepth": MAX_DEPTH,
            "featureCandidatesPerSplit": FEATURE_CANDIDATES_PER_SPLIT,
        },
        "forest": {
            "cumulativeTestAuc": rounded(
                cumulative_auc(tree_probability=tree_probability, labels=y_test)
            ),
            "trees": tree_summaries,
        },
        "predictionObservation": prediction_observation,
    }

    OUTPUT_PATH.write_text(
        "window.MECE4520RandomForestExplorer = "
        + json.dumps(payload, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
