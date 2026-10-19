"""Generate the compact, precomputed payload for the AdaBoost explorer."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


FEATURES = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
TARGET = "high_co"
RANDOM_STATE = 4520
N_ROUNDS = 12
MAX_DEPTH = 1
# This position is fixed from the reproducible test split before fitting any
# stump. It is not selected from a fitted score or prediction.
FIXED_TEST_POSITION = 4520
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "site" / "data" / "gas-turbine-course.csv"
OUTPUT_PATH = ROOT / "site" / "assets" / "adaboost-explorer.js"


def rounded(values: Sequence[float] | np.ndarray, digits: int = 5) -> list[float]:
    """Convert numeric values into compact JSON-ready floats.

    Args:
        values: Values to serialize.
        digits: Number of decimal places to retain.

    Returns:
        Rounded Python floats.
    """
    return [round(float(value), digits) for value in values]


def root_rule(*, stump: DecisionTreeClassifier) -> dict[str, object]:
    """Describe the one split used by a fitted decision stump.

    Args:
        stump: A fitted depth-one decision tree.

    Returns:
        The selected feature and its cutpoint.
    """
    feature_index = int(stump.tree_.feature[0])
    return {
        "feature": FEATURES[feature_index],
        "threshold": round(float(stump.tree_.threshold[0]), 3),
    }


def update_weights(*, weights: np.ndarray, incorrect: np.ndarray, alpha: float) -> np.ndarray:
    """Increase the emphasis on samples misclassified by one AdaBoost stump.

    Args:
        weights: Normalized sample weights before fitting the next stump.
        incorrect: Whether the current stump misclassified each training sample.
        alpha: Vote weight assigned to the current stump.

    Returns:
        Normalized row weights for the next boosting round.
    """
    updated = weights * np.exp(np.where(incorrect, alpha, -alpha))
    return updated / updated.sum()


def main() -> None:
    """Fit sequential weighted stumps and write the explorer data payload."""
    data = pd.read_csv(DATA_PATH)
    model_data = data[FEATURES + [TARGET]].dropna().copy()
    X_train, X_test, y_train, y_test = train_test_split(
        model_data[FEATURES],
        model_data[TARGET],
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=model_data[TARGET],
    )

    training_labels = y_train.to_numpy(dtype=int)
    test_labels = y_test.to_numpy(dtype=int)
    weights = np.full(len(X_train), 1 / len(X_train), dtype=float)
    cumulative_test_score = np.zeros(len(X_test), dtype=float)
    alpha_total = 0.0
    rounds: list[dict[str, object]] = []

    for round_index in range(N_ROUNDS):
        stump = DecisionTreeClassifier(
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
        )
        stump.fit(X=X_train, y=y_train, sample_weight=weights)

        train_prediction = stump.predict(X_train).astype(int)
        incorrect = train_prediction != training_labels
        weighted_error = float(weights[incorrect].sum())
        alpha = 0.5 * np.log((1 - weighted_error) / weighted_error)
        updated_weights = update_weights(
            weights=weights,
            incorrect=incorrect,
            alpha=alpha,
        )

        test_prediction = stump.predict(X_test).astype(int)
        cumulative_test_score += alpha * (2 * test_prediction - 1)
        alpha_total += alpha
        positive_vote = (cumulative_test_score / alpha_total + 1) / 2

        rounds.append(
            {
                "round": round_index + 1,
                "root": root_rule(stump=stump),
                "weightedError": round(weighted_error, 5),
                "alpha": round(float(alpha), 5),
                "incorrectRows": int(incorrect.sum()),
                "incorrectRowFraction": round(float(incorrect.mean()), 5),
                "incorrectWeightAfter": round(float(updated_weights[incorrect].sum()), 5),
                "largestWeightRelativeToUniform": round(
                    float(updated_weights.max() * len(updated_weights)), 3
                ),
                "testPrediction": int(test_prediction[FIXED_TEST_POSITION]),
                "cumulativePositiveVote": round(
                    float(positive_vote[FIXED_TEST_POSITION]), 5
                ),
                "testAuc": round(float(roc_auc_score(test_labels, positive_vote)), 5),
            }
        )
        weights = updated_weights

    fixed_row = X_test.iloc[FIXED_TEST_POSITION]
    payload = {
        "metadata": {
            "nRounds": N_ROUNDS,
            "nFeatures": len(FEATURES),
            "nTraining": int(len(X_train)),
            "nTest": int(len(X_test)),
            "maxDepth": MAX_DEPTH,
        },
        "rounds": rounds,
        "predictionObservation": {
            "actualLabel": int(y_test.iloc[FIXED_TEST_POSITION]),
            "features": {
                feature: round(float(fixed_row[feature]), 2) for feature in FEATURES
            },
        },
    }

    OUTPUT_PATH.write_text(
        "window.MECE4520AdaBoostExplorer = "
        + json.dumps(payload, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
