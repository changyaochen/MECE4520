"""Generate the compact, precomputed data payload for the RF explorer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


FEATURES = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
TARGET = "high_co"
RANDOM_STATE = 4520
N_TREES = 200
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "site" / "data" / "gas-turbine-course.csv"
OUTPUT_PATH = ROOT / "site" / "assets" / "random-forest-explorer.js"


def individual_tree_probabilities(*, model: object, X: pd.DataFrame) -> np.ndarray:
    """Collect positive-label probabilities from every tree.

    Args:
        model: A fitted ensemble whose estimators are decision trees.
        X: Observations to score.

    Returns:
        A matrix with one row per observation and one column per tree.
    """
    X_values = X.to_numpy()
    return np.column_stack(
        [tree.predict_proba(X_values)[:, 1] for tree in model.estimators_]
    )


def cumulative_auc(*, tree_probabilities: np.ndarray, labels: pd.Series) -> np.ndarray:
    """Calculate test AUC as trees are added to an ensemble.

    Args:
        tree_probabilities: Test-set probabilities from every individual tree.
        labels: Held-out binary labels.

    Returns:
        AUC values for forests containing one through all trees.
    """
    counts = np.arange(1, tree_probabilities.shape[1] + 1)
    running_probability = np.cumsum(tree_probabilities, axis=1) / counts
    return np.array(
        [
            roc_auc_score(labels, running_probability[:, index])
            for index in range(tree_probabilities.shape[1])
        ]
    )


def mean_pairwise_correlation(*, tree_probabilities: np.ndarray) -> float:
    """Summarize similarity among tree predictions.

    Args:
        tree_probabilities: Test-set probabilities from every individual tree.

    Returns:
        The mean upper-triangle correlation between individual trees.
    """
    correlation = np.corrcoef(tree_probabilities, rowvar=False)
    indices = np.triu_indices_from(correlation, k=1)
    return float(correlation[indices].mean())


def root_features(*, model: object) -> list[str]:
    """Return the feature selected at the root of each tree.

    Args:
        model: A fitted ensemble whose estimators are decision trees.

    Returns:
        Root-split feature names, in ensemble order.
    """
    return [FEATURES[int(tree.tree_.feature[0])] for tree in model.estimators_]


def choose_position(
    *,
    probability: np.ndarray,
    labels: pd.Series,
    target_probability: float,
    target_label: int,
) -> int:
    """Find one held-out observation near a requested probability and label.

    Args:
        probability: Final forest probabilities on the test set.
        labels: Test-set binary labels.
        target_probability: Desired final forest probability.
        target_label: Required observed label.

    Returns:
        Positional index of the selected held-out observation.
    """
    eligible_positions = np.flatnonzero(labels.to_numpy() == target_label)
    closest = np.argmin(
        np.abs(probability[eligible_positions] - target_probability)
    )
    return int(eligible_positions[closest])


def rounded(values: np.ndarray, digits: int = 5) -> list[float]:
    """Convert a numeric array into compact JSON-friendly values.

    Args:
        values: Numeric values to serialize.
        digits: Number of decimal places to retain.

    Returns:
        Rounded Python floats.
    """
    return [round(float(value), digits) for value in values]


def main() -> None:
    """Fit the two ensembles and write their explorer data as JavaScript."""
    data = pd.read_csv(DATA_PATH)
    model_data = data[FEATURES + [TARGET]].dropna().copy()
    X_train, X_test, y_train, y_test = train_test_split(
        model_data[FEATURES],
        model_data[TARGET],
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=model_data[TARGET],
    )

    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=N_TREES,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    forest = RandomForestClassifier(
        n_estimators=N_TREES,
        max_features="sqrt",
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    bagging.fit(X=X_train, y=y_train)
    forest.fit(X=X_train, y=y_train)

    bagging_tree_probability = individual_tree_probabilities(
        model=bagging, X=X_test
    )
    forest_tree_probability = individual_tree_probabilities(model=forest, X=X_test)
    forest_probability = forest.predict_proba(X_test)[:, 1]

    positions = [
        (
            "clearly-negative",
            "Clearly negative",
            choose_position(
                probability=forest_probability,
                labels=y_test,
                target_probability=0.02,
                target_label=0,
            ),
        ),
        (
            "borderline",
            "Borderline",
            choose_position(
                probability=forest_probability,
                labels=y_test,
                target_probability=0.50,
                target_label=0,
            ),
        ),
        (
            "clearly-positive",
            "Clearly positive",
            choose_position(
                probability=forest_probability,
                labels=y_test,
                target_probability=0.98,
                target_label=1,
            ),
        ),
    ]

    selected_features = ["TIT", "CDP", "TEY"]
    observations = []
    for observation_id, name, position in positions:
        row = X_test.iloc[position]
        observations.append(
            {
                "id": observation_id,
                "name": name,
                "actualLabel": int(y_test.iloc[position]),
                "features": {
                    feature: round(float(row[feature]), 2)
                    for feature in selected_features
                },
                "models": {
                    "bagging": {
                        "treeProbabilities": rounded(
                            bagging_tree_probability[position]
                        )
                    },
                    "randomForest": {
                        "treeProbabilities": rounded(
                            forest_tree_probability[position]
                        )
                    },
                },
            }
        )

    payload = {
        "metadata": {
            "nTrees": N_TREES,
            "nTraining": int(len(X_train)),
            "nTest": int(len(X_test)),
            "positiveFraction": round(float(y_train.mean()), 4),
            "featureCandidatesPerSplit": 3,
        },
        "models": {
            "bagging": {
                "label": "Bagging",
                "auc": rounded(
                    cumulative_auc(
                        tree_probabilities=bagging_tree_probability, labels=y_test
                    )
                ),
                "roots": root_features(model=bagging),
                "meanPairwiseCorrelation": round(
                    mean_pairwise_correlation(
                        tree_probabilities=bagging_tree_probability
                    ),
                    4,
                ),
            },
            "randomForest": {
                "label": "Random forest",
                "auc": rounded(
                    cumulative_auc(
                        tree_probabilities=forest_tree_probability, labels=y_test
                    )
                ),
                "roots": root_features(model=forest),
                "meanPairwiseCorrelation": round(
                    mean_pairwise_correlation(
                        tree_probabilities=forest_tree_probability
                    ),
                    4,
                ),
            },
        },
        "observations": observations,
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
