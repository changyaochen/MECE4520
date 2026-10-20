"""Generate a compact, precomputed payload for the gradient-boosting explorer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


FEATURES = ["AT", "GTEP", "TIT", "TAT", "TEY", "CDP"]
TARGET = "NOX"
RANDOM_STATE = 4520
N_STAGES = 60
LEARNING_RATE = 0.1
MAX_DEPTH = 2
MIN_SAMPLES_LEAF = 50
RESIDUAL_HISTOGRAM_BINS = 32
# Fixed illustrative held-out observation used throughout the explorer.
FIXED_TEST_POSITION = 133
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "site" / "data" / "gas-turbine-course.csv"
OUTPUT_PATH = ROOT / "site" / "assets" / "gradient-boosting-explorer.js"


def tree_leaves(*, tree: DecisionTreeRegressor) -> list[dict[str, object]]:
    """Describe each terminal region of a fitted shallow regression tree.

    Args:
        tree: Fitted regression tree that supplies an additive target correction.

    Returns:
        One dictionary per leaf with its rule, correction, and training count.
    """
    structure = tree.tree_
    leaves: list[dict[str, object]] = []

    def visit(*, node: int, conditions: list[str]) -> None:
        """Traverse the tree and record terminal-region rules.

        Args:
            node: Node currently being visited.
            conditions: Rules needed to reach the node.
        """
        left_child = int(structure.children_left[node])
        right_child = int(structure.children_right[node])
        if left_child == right_child:
            leaves.append(
                {
                    "rule": " and ".join(conditions),
                    "correction": round(float(structure.value[node, 0, 0]), 5),
                    "nSamples": int(structure.n_node_samples[node]),
                }
            )
            return

        feature = FEATURES[int(structure.feature[node])]
        threshold = float(structure.threshold[node])
        visit(
            node=left_child,
            conditions=[*conditions, f"{feature} ≤ {threshold:.3f}"],
        )
        visit(
            node=right_child,
            conditions=[*conditions, f"{feature} > {threshold:.3f}"],
        )

    visit(node=0, conditions=[])
    return leaves


def matching_leaf(*, leaves: list[dict[str, object]], correction: float) -> str:
    """Find the displayed leaf rule associated with a tree correction.

    Args:
        leaves: Terminal-region descriptions from tree_leaves.
        correction: Regression-tree output for one observation.

    Returns:
        Rule displayed for the matching leaf.
    """
    for leaf in leaves:
        if np.isclose(float(leaf["correction"]), correction, atol=1e-5):
            return str(leaf["rule"])
    return "Selected terminal region"


def mse(*, observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute mean squared error as a Python float.

    Args:
        observed: Measured target values.
        predicted: Model predictions.

    Returns:
        Mean squared error.
    """
    return float(mean_squared_error(y_true=observed, y_pred=predicted))


def main() -> None:
    """Fit staged gradient boosting and write data used by the explorer."""
    data = pd.read_csv(DATA_PATH)
    model_data = data[FEATURES + [TARGET]].dropna().copy()
    X_train, X_test, y_train, y_test = train_test_split(
        model_data[FEATURES],
        model_data[TARGET],
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    model = GradientBoostingRegressor(
        loss="squared_error",
        n_estimators=N_STAGES,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )
    model.fit(X=X_train, y=y_train)

    train_targets = y_train.to_numpy(dtype=float)
    test_targets = y_test.to_numpy(dtype=float)
    baseline_prediction = float(train_targets.mean())
    baseline_train_predictions = np.full(
        len(X_train),
        baseline_prediction,
        dtype=float,
    )
    baseline_test_predictions = np.full(
        len(X_test),
        baseline_prediction,
        dtype=float,
    )
    train_predictions = baseline_train_predictions.copy()
    test_predictions = baseline_test_predictions.copy()

    rounds: list[dict[str, object]] = []
    residuals_by_stage: list[np.ndarray] = []
    train_values = X_train.to_numpy()
    test_values = X_test.to_numpy()

    for stage_index in range(N_STAGES):
        correction_tree = model.estimators_[stage_index, 0]
        residuals = train_targets - train_predictions
        residuals_by_stage.append(residuals.copy())
        tree_output_train = correction_tree.predict(train_values)
        tree_output_test = correction_tree.predict(test_values)
        leaves = tree_leaves(tree=correction_tree)

        fixed_correction = float(tree_output_test[FIXED_TEST_POSITION])
        fixed_prediction_before = float(test_predictions[FIXED_TEST_POSITION])

        train_predictions += LEARNING_RATE * tree_output_train
        test_predictions += LEARNING_RATE * tree_output_test

        root_feature_index = int(correction_tree.tree_.feature[0])
        rounds.append(
            {
                "stage": stage_index + 1,
                "root": {
                    "feature": FEATURES[root_feature_index],
                    "threshold": round(float(correction_tree.tree_.threshold[0]), 3),
                },
                "leaves": leaves,
                "fixedObservation": {
                    "predictionBefore": round(fixed_prediction_before, 5),
                    "treeCorrection": round(fixed_correction, 5),
                    "predictionChange": round(
                        LEARNING_RATE * fixed_correction,
                        5,
                    ),
                    "predictionAfter": round(
                        float(test_predictions[FIXED_TEST_POSITION]),
                        5,
                    ),
                    "leafRule": matching_leaf(
                        leaves=leaves,
                        correction=fixed_correction,
                    ),
                },
                "trainMse": round(
                    mse(observed=train_targets, predicted=train_predictions),
                    5,
                ),
                "testMse": round(
                    mse(observed=test_targets, predicted=test_predictions),
                    5,
                ),
                "testR2": round(
                    float(r2_score(y_true=test_targets, y_pred=test_predictions)),
                    5,
                ),
            }
        )

    residual_limit = float(
        np.ceil(
            max(float(np.abs(residuals).max()) for residuals in residuals_by_stage)
            / 5.0
        )
        * 5.0
    )
    residual_bin_edges = np.linspace(
        -residual_limit,
        residual_limit,
        RESIDUAL_HISTOGRAM_BINS + 1,
    )
    residual_histogram_max_count = 0
    for round_data, residuals in zip(rounds, residuals_by_stage, strict=True):
        counts, _ = np.histogram(residuals, bins=residual_bin_edges)
        residual_histogram_max_count = max(
            residual_histogram_max_count,
            int(counts.max()),
        )
        round_data["residualHistogram"] = counts.tolist()

    final_model_prediction = model.predict(X_test)
    if not np.allclose(test_predictions, final_model_prediction):
        raise RuntimeError(
            "Manual stage-wise predictions do not match GradientBoostingRegressor."
        )

    fixed_row = X_test.iloc[FIXED_TEST_POSITION]
    payload = {
        "metadata": {
            "nStages": N_STAGES,
            "nFeatures": len(FEATURES),
            "nTraining": int(len(X_train)),
            "nTest": int(len(X_test)),
            "learningRate": LEARNING_RATE,
            "maxDepth": MAX_DEPTH,
            "minSamplesLeaf": MIN_SAMPLES_LEAF,
            "targetName": "NOx",
            "targetUnit": "mg/m³",
            "residualHistogram": {
                "binEdges": [round(float(edge), 3) for edge in residual_bin_edges],
                "maxCount": residual_histogram_max_count,
            },
        },
        "baseline": {
            "prediction": round(baseline_prediction, 5),
            "trainMse": round(
                mse(
                    observed=train_targets,
                    predicted=baseline_train_predictions,
                ),
                5,
            ),
            "testMse": round(
                mse(
                    observed=test_targets,
                    predicted=baseline_test_predictions,
                ),
                5,
            ),
            "testR2": round(
                float(
                    r2_score(
                        y_true=test_targets,
                        y_pred=baseline_test_predictions,
                    )
                ),
                5,
            ),
        },
        "rounds": rounds,
        "predictionObservation": {
            "observedTarget": round(
                float(test_targets[FIXED_TEST_POSITION]),
                2,
            ),
            "features": {
                feature: round(float(fixed_row[feature]), 2) for feature in FEATURES
            },
        },
    }

    OUTPUT_PATH.write_text(
        "window.MECE4520GradientBoostingExplorer = "
        + json.dumps(payload, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
