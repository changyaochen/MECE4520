"""Generate compact, precomputed data for the Silverbox AR--ARX--RNN explorer.

The script deliberately keeps the raw Silverbox benchmark outside this
repository.  Install ``nonlinear-benchmarks`` first; its ``Silverbox`` loader
downloads the official benchmark archive linked from nonlinearbenchmark.org.

The linear models use a fixed 20-sample response-lag initialization window.
The RNN uses those same first 20 held-out responses to prime its carried
hidden state.  Afterwards, all three models forecast the output recursively.
The applied input remains available at every held-out time step, which is the
usual system-identification setting for a known forcing signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


SOURCE_URL = "https://www.nonlinearbenchmark.org/benchmarks/silverbox"
ORDER = 20
RANDOM_SEED = 7
HIDDEN_SIZE = 16
EPOCHS = 50
CHUNK_SIZE = 256
SEGMENT_START = 6_000
SEGMENT_LENGTH = 1_500


class RecurrentPredictor(torch.nn.Module):
    """A small vanilla RNN followed by a nonlinear scalar readout."""

    def __init__(self, *, hidden_size: int) -> None:
        """Initialize the recurrent state and output head.

        Args:
            hidden_size: Number of learned values in the recurrent state.
        """
        super().__init__()
        self.rnn = torch.nn.RNN(
            input_size=2,
            hidden_size=hidden_size,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict responses and return the state for the next sequence."""
        hidden_values, next_hidden_state = self.rnn(sequence, hidden_state)
        predictions = self.readout(hidden_values).squeeze(-1)
        return predictions, next_hidden_state


def import_silverbox_loader():
    """Return the official-loader function with a useful setup message."""
    try:
        from nonlinear_benchmarks import Silverbox
    except ImportError as error:
        message = (
            "This generator needs the official benchmark loader. Install it with\n"
            "    pip install nonlinear-benchmarks\n"
            "Then run this script again. The loader downloads the official "
            "Silverbox archive into its local cache; raw benchmark data are not "
            "committed to this repository."
        )
        raise SystemExit(message) from error
    return Silverbox


def make_linear_design(
    *,
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    order: int,
    include_input: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one-step AR or ARX design rows from a single input-output record."""
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for index in range(order, len(output_signal)):
        output_lags = output_signal[index - order : index][::-1]
        if include_input:
            input_lags = input_signal[index - order : index + 1][::-1]
            rows.append(np.concatenate((output_lags, input_lags)))
        else:
            rows.append(output_lags)
        targets.append(output_signal[index])
    return np.asarray(rows), np.asarray(targets)


def fit_linear_model(
    *,
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    order: int,
    include_input: bool,
) -> tuple[StandardScaler, LinearRegression]:
    """Fit an ordinary-least-squares linear AR or ARX predictor."""
    features, targets = make_linear_design(
        input_signal=input_signal,
        output_signal=output_signal,
        order=order,
        include_input=include_input,
    )
    scaler = StandardScaler().fit(features)
    model = LinearRegression().fit(scaler.transform(features), targets)
    return scaler, model


def recursive_linear_forecast(
    *,
    input_signal: np.ndarray,
    observed_output: np.ndarray,
    order: int,
    include_input: bool,
    scaler: StandardScaler,
    model: LinearRegression,
) -> np.ndarray:
    """Forecast a test record after using only its first ``order`` outputs.

    The forcing signal is known throughout the test record.  After the warmup
    period, the output-lag portion of each input row contains prior forecasts,
    never later observed outputs.
    """
    history = list(observed_output[:order])
    predictions = np.full(len(observed_output), np.nan)
    for index in range(order, len(observed_output)):
        output_lags = np.asarray(history[-order:][::-1])
        if include_input:
            input_lags = input_signal[index - order : index + 1][::-1]
            row = np.concatenate((output_lags, input_lags))
        else:
            row = output_lags
        prediction = model.predict(scaler.transform(row.reshape(1, -1)))[0]
        history.append(prediction)
        predictions[index] = prediction
    return predictions


def make_teacher_forced_sequence(
    *,
    input_signal: np.ndarray,
    output_signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create chronological ``[forcing, previous response]`` inputs and targets."""
    inputs = np.column_stack((input_signal[1:], output_signal[:-1])).astype(np.float32)
    targets = output_signal[1:].astype(np.float32)
    return inputs, targets


def detach_recurrent_state(hidden_state: torch.Tensor) -> torch.Tensor:
    """Keep state values but remove their old gradient history."""
    return hidden_state.detach()


def fit_rnn(
    *,
    input_signal: np.ndarray,
    output_signal: np.ndarray,
) -> tuple[RecurrentPredictor, tuple[float, float, float, float]]:
    """Fit a state-carrying RNN with chronological teacher-forced chunks."""
    input_mean = float(input_signal.mean())
    input_std = float(input_signal.std())
    output_mean = float(output_signal.mean())
    output_std = float(output_signal.std())
    inputs_scaled = (input_signal - input_mean) / input_std
    outputs_scaled = (output_signal - output_mean) / output_std
    inputs, targets = make_teacher_forced_sequence(
        input_signal=inputs_scaled,
        output_signal=outputs_scaled,
    )

    torch.manual_seed(RANDOM_SEED)
    model = RecurrentPredictor(hidden_size=HIDDEN_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()

    for _ in range(EPOCHS):
        model.train()
        hidden_state = None
        for start in range(0, len(inputs), CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, len(inputs))
            sequence_chunk = torch.from_numpy(inputs[start:stop][None, ...])
            target_chunk = torch.from_numpy(targets[start:stop][None, ...])
            optimizer.zero_grad()
            predictions, hidden_state = model(sequence_chunk, hidden_state)
            loss = criterion(predictions, target_chunk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            hidden_state = detach_recurrent_state(hidden_state)

    return model, (input_mean, input_std, output_mean, output_std)


def recursive_rnn_forecast(
    *,
    model: RecurrentPredictor,
    input_signal: np.ndarray,
    observed_output: np.ndarray,
    warmup: int,
    scales: tuple[float, float, float, float],
) -> np.ndarray:
    """Prime an RNN state, then recursively forecast with known input values."""
    input_mean, input_std, output_mean, output_std = scales
    predictions = np.full(len(observed_output), np.nan)
    model.eval()
    with torch.no_grad():
        hidden_state = None
        for index in range(1, warmup):
            sequence = np.array(
                [[[
                    (input_signal[index] - input_mean) / input_std,
                    (observed_output[index - 1] - output_mean) / output_std,
                ]]],
                dtype=np.float32,
            )
            _, hidden_state = model(torch.from_numpy(sequence), hidden_state)

        previous_response = float(observed_output[warmup - 1])
        for index in range(warmup, len(observed_output)):
            sequence = np.array(
                [[[
                    (input_signal[index] - input_mean) / input_std,
                    (previous_response - output_mean) / output_std,
                ]]],
                dtype=np.float32,
            )
            standardized_prediction, hidden_state = model(
                torch.from_numpy(sequence), hidden_state
            )
            prediction = standardized_prediction * output_std + output_mean
            previous_response = prediction.item()
            predictions[index] = previous_response
    return predictions


def rmse(*, actual: np.ndarray, predicted: np.ndarray, warmup: int) -> float:
    """Compute recursive multi-step test root-mean-square error after warmup."""
    return float(np.sqrt(np.mean((actual[warmup:] - predicted[warmup:]) ** 2)))


def rounded(values: np.ndarray, *, decimals: int = 6) -> list[float | None]:
    """Convert numerical values to compact JSON-friendly values."""
    return [None if not np.isfinite(value) else round(float(value), decimals) for value in values]


def parse_arguments() -> argparse.Namespace:
    """Read the optional output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("site/assets/silverbox-explorer-data.js"),
        help="JavaScript asset to create.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the benchmark, fit the three models, and write compact explorer data."""
    arguments = parse_arguments()
    torch.set_num_threads(4)
    silverbox = import_silverbox_loader()
    training, tests = silverbox()
    test = tests[0]  # Official held-out multisine record.
    input_train = np.asarray(training.u, dtype=float).reshape(-1)
    output_train = np.asarray(training.y, dtype=float).reshape(-1)
    input_test = np.asarray(test.u, dtype=float).reshape(-1)
    output_test = np.asarray(test.y, dtype=float).reshape(-1)

    ar_scaler, ar_model = fit_linear_model(
        input_signal=input_train,
        output_signal=output_train,
        order=ORDER,
        include_input=False,
    )
    arx_scaler, arx_model = fit_linear_model(
        input_signal=input_train,
        output_signal=output_train,
        order=ORDER,
        include_input=True,
    )
    ar_predictions = recursive_linear_forecast(
        input_signal=input_test,
        observed_output=output_test,
        order=ORDER,
        include_input=False,
        scaler=ar_scaler,
        model=ar_model,
    )
    arx_predictions = recursive_linear_forecast(
        input_signal=input_test,
        observed_output=output_test,
        order=ORDER,
        include_input=True,
        scaler=arx_scaler,
        model=arx_model,
    )
    rnn_model, scales = fit_rnn(
        input_signal=input_train,
        output_signal=output_train,
    )
    rnn_predictions = recursive_rnn_forecast(
        model=rnn_model,
        input_signal=input_test,
        observed_output=output_test,
        warmup=ORDER,
        scales=scales,
    )

    segment_start = min(SEGMENT_START, len(output_test) - SEGMENT_LENGTH)
    segment_end = segment_start + SEGMENT_LENGTH
    indices = np.arange(segment_start, segment_end)
    payload = {
        "sourceUrl": SOURCE_URL,
        "samplingTime": round(float(training.sampling_time), 9),
        "order": ORDER,
        "trainSamples": int(len(output_train)),
        "testSamples": int(len(output_test)),
        "warmup": ORDER,
        "forecastProtocol": {
            "initialization": (
                "The first 20 held-out responses initialize the AR and ARX lags "
                "and prime the RNN state."
            ),
            "afterInitialization": (
                "AR and ARX reuse their own forecast lags; the RNN carries its "
                "hidden state and feeds back its prior forecast. The forcing input "
                "remains known."
            ),
        },
        "metrics": {
            "ar": {
                "label": "AR(20)",
                "rmse": round(rmse(actual=output_test, predicted=ar_predictions, warmup=ORDER), 6),
                "uses": "20 earlier response values",
            },
            "arx": {
                "label": "ARX(20)",
                "rmse": round(rmse(actual=output_test, predicted=arx_predictions, warmup=ORDER), 6),
                "uses": "20 earlier response values + known forcing input",
            },
            "rnn": {
                "label": "Vanilla RNN",
                "rmse": round(rmse(actual=output_test, predicted=rnn_predictions, warmup=ORDER), 6),
                "uses": "known forcing input + prior response + carried nonlinear state",
            },
        },
        "segment": {
            "start": int(segment_start),
            "end": int(segment_end - 1),
            "timeMilliseconds": rounded(indices * float(training.sampling_time) * 1_000, decimals=4),
            "input": rounded(input_test[segment_start:segment_end]),
            "observed": rounded(output_test[segment_start:segment_end]),
            "predictions": {
                "ar": rounded(ar_predictions[segment_start:segment_end]),
                "arx": rounded(arx_predictions[segment_start:segment_end]),
                "rnn": rounded(rnn_predictions[segment_start:segment_end]),
            },
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        "window.SILVERBOX_EXPLORER_DATA = "
        + json.dumps(payload, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {arguments.output}")
    for key, details in payload["metrics"].items():
        print(f"{details['label']}: recursive multi-step test RMSE = {details['rmse']:.6f}")


if __name__ == "__main__":
    main()
