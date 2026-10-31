"""Create the compact ETT-small payload used by the AR-order explorer."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ORDERS = (1, 6, 24, 168)
TRAIN_END = 700
SOURCE_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"


def fit_ar_model(*, values: np.ndarray, order: int) -> tuple[float, np.ndarray]:
    """Fit an ordinary least-squares AR(order) model on the training values."""
    design = np.array([values[index - order:index][::-1] for index in range(order, len(values))])
    target = values[order:]
    design_with_intercept = np.column_stack([np.ones(len(design)), design])
    solution, *_ = np.linalg.lstsq(design_with_intercept, target, rcond=None)
    return float(solution[0]), solution[1:]


def remove_repeated_value_artifact(*, frame: pd.DataFrame) -> pd.DataFrame:
    """Remove the 24-hour repeated OT value from the displayed time window."""
    run_id = frame["OT"].ne(frame["OT"].shift()).cumsum()
    run_length = frame.groupby(run_id)["OT"].transform("size")
    return frame.loc[run_length < 24].copy()


def main() -> None:
    """Fit four AR models and write their compact held-out payload."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional local copy of ETTh1.csv; otherwise download the public source.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("site/assets/ar-order-explorer-data.js"),
        help="Explorer JavaScript payload to write.",
    )
    arguments = parser.parse_args()
    source = arguments.input if arguments.input is not None else SOURCE_URL
    frame = pd.read_csv(source, on_bad_lines="skip").dropna(subset=["OT"]).iloc[:1000]
    frame = remove_repeated_value_artifact(frame=frame)
    values = frame["OT"].to_numpy(dtype=float)
    dates = pd.to_datetime(frame["date"]).dt.strftime("%b %-d, %H:%M").tolist()

    models: dict[str, dict[str, object]] = {}
    for order in ORDERS:
        intercept, coefficients = fit_ar_model(values=values[:TRAIN_END], order=order)
        one_step_predictions: list[float | None] = [None] * TRAIN_END
        for index in range(TRAIN_END, len(values)):
            prediction = intercept + np.dot(coefficients, values[index - order:index][::-1])
            one_step_predictions.append(round(float(prediction), 3))

        recursive_predictions: list[float | None] = [None] * TRAIN_END
        history = list(values[:TRAIN_END])
        for _ in range(TRAIN_END, len(values)):
            prediction = intercept + np.dot(coefficients, history[-order:][::-1])
            recursive_predictions.append(round(float(prediction), 3))
            history.append(prediction)
        one_step_errors = values[TRAIN_END:] - np.asarray(one_step_predictions[TRAIN_END:], dtype=float)
        recursive_errors = values[TRAIN_END:] - np.asarray(recursive_predictions[TRAIN_END:], dtype=float)
        models[str(order)] = {
            "intercept": round(intercept, 6),
            "coefficients": [round(float(value), 6) for value in coefficients],
            "oneStepPredictions": one_step_predictions,
            "recursivePredictions": recursive_predictions,
            "oneStepRmse": round(float(np.sqrt(np.mean(one_step_errors**2))), 3),
            "recursiveRmse": round(float(np.sqrt(np.mean(recursive_errors**2))), 3),
        }

    payload = {
        "source": "ETT-small ETTh1 hourly oil-temperature series",
        "target": "OT",
        "trainEnd": TRAIN_END,
        "dates": dates,
        "values": [round(float(value), 3) for value in values],
        "models": models,
    }
    arguments.output.write_text(
        "window.AR_ORDER_EXPLORER_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
