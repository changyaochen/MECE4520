(() => {
  const root = document.getElementById("bptt-explorer");
  if (!root) return;

  const activationInput = root.querySelector("#bptt-activation");
  const weightInput = root.querySelector("#bptt-weight");
  const lengthInput = root.querySelector("#bptt-length");
  const rateInput = root.querySelector("#bptt-rate");
  const weightValue = root.querySelector("#bptt-weight-value");
  const lengthValue = root.querySelector("#bptt-length-value");
  const rateValue = root.querySelector("#bptt-rate-value");
  const activationEquation = root.querySelector("#bptt-activation-equation");
  const matrices = root.querySelector("#bptt-matrices");
  const unrollSvg = root.querySelector("#bptt-unroll");
  const stateChart = root.querySelector("#bptt-state-chart");
  const gradientChart = root.querySelector("#bptt-gradient-chart");
  const metrics = root.querySelector("#bptt-metrics");
  const summaryCopy = root.querySelector("#bptt-summary-copy");
  const tableBody = root.querySelector("#bptt-table-body");
  const status = root.querySelector("#bptt-status");
  const stepButton = root.querySelector("#bptt-step");
  const resetButton = root.querySelector("#bptt-reset");
  const presetButtons = [...root.querySelectorAll("[data-bptt-preset]")];

  if (!activationInput || !weightInput || !lengthInput || !rateInput || !weightValue
      || !lengthValue || !rateValue || !activationEquation || !matrices || !unrollSvg
      || !stateChart || !gradientChart || !metrics || !summaryCopy || !tableBody || !status
      || !stepButton || !resetButton || !presetButtons.length) return;

  const SVG_NS = "http://www.w3.org/2000/svg";
  const INPUT_VECTOR = [0.35, 0.15];
  const OUTPUT_VECTOR = [1, 0];
  const TARGET = 1;
  const ROTATION_ANGLE = 10 * Math.PI / 180;
  const BASE_ROTATION = [
    [Math.cos(ROTATION_ANGLE), -Math.sin(ROTATION_ANGLE)],
    [Math.sin(ROTATION_ANGLE), Math.cos(ROTATION_ANGLE)],
  ];
  const SCALE_LIMITS = { min: -1.5, max: 1.5 };
  const PRESETS = {
    vanishing: {
      activation: "linear",
      scale: 0.70,
      length: 16,
      rate: 0.01,
      status: "Vanishing-gradient setup: each linear backward step shrinks the signal.",
    },
    balanced: {
      activation: "linear",
      scale: 1.00,
      length: 16,
      rate: 0.01,
      status: "Balanced setup: the base rotation preserves gradient-vector norm.",
    },
    exploding: {
      activation: "linear",
      scale: 1.30,
      length: 16,
      rate: 0.001,
      status: "Exploding-gradient setup: each linear backward step amplifies the signal.",
    },
  };
  const state = { ...PRESETS.vanishing, activePreset: "vanishing" };

  function zeroVector() {
    return [0, 0];
  }

  function zeroMatrix() {
    return [[0, 0], [0, 0]];
  }

  function scaleMatrix(matrix, scale) {
    return matrix.map((row) => row.map((value) => scale * value));
  }

  function matrixVector(matrix, vector) {
    return matrix.map((row) => row[0] * vector[0] + row[1] * vector[1]);
  }

  function transposeMatrixVector(matrix, vector) {
    return [
      matrix[0][0] * vector[0] + matrix[1][0] * vector[1],
      matrix[0][1] * vector[0] + matrix[1][1] * vector[1],
    ];
  }

  function vectorAdd(left, right) {
    return [left[0] + right[0], left[1] + right[1]];
  }

  function hadamard(left, right) {
    return [left[0] * right[0], left[1] * right[1]];
  }

  function outerProduct(left, right) {
    return [
      [left[0] * right[0], left[0] * right[1]],
      [left[1] * right[0], left[1] * right[1]],
    ];
  }

  function addMatrixInPlace(target, source) {
    for (let row = 0; row < 2; row += 1) {
      for (let column = 0; column < 2; column += 1) {
        target[row][column] += source[row][column];
      }
    }
  }

  function vectorNorm(vector) {
    return Math.hypot(...vector);
  }

  function matrixNorm(matrix) {
    return Math.hypot(...matrix.flat());
  }

  function frobeniusInner(left, right) {
    return left.flat().reduce((sum, value, index) => sum + value * right.flat()[index], 0);
  }

  function applyActivation(vector) {
    return state.activation === "tanh"
      ? vector.map((value) => Math.tanh(value))
      : [...vector];
  }

  function activationDerivative(hidden) {
    return state.activation === "tanh"
      ? hidden.map((value) => 1 - value ** 2)
      : [1, 1];
  }

  function simulate() {
    const length = state.length;
    const inputs = Array(length + 1).fill(0);
    const scores = Array.from({ length: length + 1 }, zeroVector);
    const hidden = Array.from({ length: length + 1 }, zeroVector);
    const derivatives = Array.from({ length: length + 1 }, zeroVector);
    const gradientHidden = Array.from({ length: length + 1 }, zeroVector);
    const delta = Array.from({ length: length + 1 }, zeroVector);
    const contributions = Array.from({ length: length + 1 }, zeroMatrix);
    const matrix = scaleMatrix(BASE_ROTATION, state.scale);

    inputs[1] = 1;
    for (let time = 1; time <= length; time += 1) {
      const recurrentPart = matrixVector(matrix, hidden[time - 1]);
      const inputPart = INPUT_VECTOR.map((value) => value * inputs[time]);
      scores[time] = vectorAdd(recurrentPart, inputPart);
      hidden[time] = applyActivation(scores[time]);
      derivatives[time] = activationDerivative(hidden[time]);
    }

    const prediction = OUTPUT_VECTOR[0] * hidden[length][0] + OUTPUT_VECTOR[1] * hidden[length][1];
    const loss = 0.5 * (prediction - TARGET) ** 2;
    gradientHidden[length] = OUTPUT_VECTOR.map((value) => (prediction - TARGET) * value);
    const gradientMatrix = zeroMatrix();

    for (let time = length; time >= 1; time -= 1) {
      delta[time] = hadamard(gradientHidden[time], derivatives[time]);
      contributions[time] = outerProduct(delta[time], hidden[time - 1]);
      addMatrixInPlace(gradientMatrix, contributions[time]);
      if (time > 1) {
        gradientHidden[time - 1] = vectorAdd(
          gradientHidden[time - 1],
          transposeMatrixVector(matrix, delta[time]),
        );
      }
    }

    return {
      inputs,
      scores,
      hidden,
      derivatives,
      gradientHidden,
      delta,
      contributions,
      matrix,
      prediction,
      loss,
      gradientMatrix,
      gradientScale: frobeniusInner(gradientMatrix, BASE_ROTATION),
    };
  }

  function createSvg(name, attributes = {}) {
    const element = document.createElementNS(SVG_NS, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
    return element;
  }

  function appendSvg(parent, name, attributes = {}, text = null) {
    const element = createSvg(name, attributes);
    if (text !== null) element.textContent = text;
    parent.append(element);
    return element;
  }

  function clear(element) {
    element.replaceChildren();
  }

  function format(value, digits = 3) {
    if (!Number.isFinite(value)) return "—";
    const magnitude = Math.abs(value);
    if (magnitude !== 0 && (magnitude < 0.001 || magnitude >= 1000)) {
      return value.toExponential(2);
    }
    return value.toFixed(digits);
  }

  function formatVector(vector, digits = 2) {
    return "[" + format(vector[0], digits) + ", " + format(vector[1], digits) + "]";
  }

  function logMagnitude(value) {
    return Math.log10(Math.max(value, 1e-12));
  }

  function setControls() {
    activationInput.value = state.activation;
    weightInput.value = state.scale;
    lengthInput.value = state.length;
    rateInput.value = state.rate;
    weightValue.textContent = format(state.scale, 2);
    lengthValue.textContent = String(state.length);
    rateValue.textContent = format(state.rate, 3);
    activationEquation.innerHTML = state.activation === "tanh"
      ? "<i>h</i><sub>t</sub> = tanh(<i>a</i><sub>t</sub>)"
      : "<i>h</i><sub>t</sub> = <i>a</i><sub>t</sub>";
    presetButtons.forEach((button) => {
      const selected = button.dataset.bpttPreset === state.activePreset;
      button.classList.toggle("is-active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
  }

  function makeMatrixGrid(matrix) {
    const grid = document.createElement("div");
    grid.className = "bptt-matrix-grid";
    matrix.flat().forEach((value) => {
      const cell = document.createElement("span");
      cell.textContent = format(value, 3);
      grid.append(cell);
    });
    return grid;
  }

  function renderMatrices(data) {
    const cards = [
      {
        label: "Fixed rotation R",
        detail: "10° rotation; ||R||₂ = 1",
        matrix: BASE_ROTATION,
        className: "bptt-matrix-base",
      },
      {
        label: "Recurrent matrix Wₕ = αR",
        detail: "α = " + format(state.scale, 2),
        matrix: data.matrix,
        className: "bptt-matrix-current",
      },
      {
        label: "Gradient ∂L/∂Wₕ",
        detail: "||∇WₕL||F = " + format(matrixNorm(data.gradientMatrix)),
        matrix: data.gradientMatrix,
        className: "bptt-matrix-gradient",
      },
    ];
    matrices.replaceChildren(...cards.map((entry) => {
      const card = document.createElement("div");
      card.className = "bptt-matrix-card " + entry.className;
      const heading = document.createElement("strong");
      heading.textContent = entry.label;
      const detail = document.createElement("small");
      detail.textContent = entry.detail;
      card.append(heading, makeMatrixGrid(entry.matrix), detail);
      return card;
    }));
  }

  function makeMarker(definitions, id, color) {
    const marker = appendSvg(definitions, "marker", {
      id,
      viewBox: "0 0 10 10",
      refX: "8",
      refY: "5",
      markerWidth: "5",
      markerHeight: "5",
      orient: "auto-start-reverse",
    });
    appendSvg(marker, "path", { d: "M 0 0 L 10 5 L 0 10 z", fill: color });
  }

  function drawUnroll(data) {
    clear(unrollSvg);
    const cellWidth = 108;
    const width = Math.max(980, 100 + state.length * cellWidth);
    const height = 318;
    const yInput = 48;
    const yState = 142;
    const yBackward = 258;
    unrollSvg.setAttribute("viewBox", "0 0 " + width + " " + height);
    unrollSvg.setAttribute("width", String(width));
    unrollSvg.setAttribute("height", String(height));

    const definitions = appendSvg(unrollSvg, "defs");
    makeMarker(definitions, "bptt-forward-arrow", "#4f7d96");
    makeMarker(definitions, "bptt-backward-arrow", "#cf6c37");
    appendSvg(unrollSvg, "text", { x: 18, y: yInput + 5, class: "bptt-svg-label" }, "input");
    appendSvg(unrollSvg, "text", { x: 18, y: yState + 5, class: "bptt-svg-label" }, "state");
    appendSvg(unrollSvg, "text", { x: 18, y: yBackward + 5, class: "bptt-svg-label" }, "BPTT");

    const gradientLogs = data.gradientHidden.slice(1).map((vector) => logMagnitude(vectorNorm(vector)));
    const minimumLog = Math.min(...gradientLogs);
    const maximumLog = Math.max(...gradientLogs);
    const normalizedGradient = (vector) => {
      if (maximumLog - minimumLog < 0.2) return 0.5;
      return (logMagnitude(vectorNorm(vector)) - minimumLog) / (maximumLog - minimumLog);
    };

    for (let time = 1; time <= state.length; time += 1) {
      const x = 76 + (time - 1) * cellWidth;
      const inputValue = data.inputs[time];
      appendSvg(unrollSvg, "rect", {
        x: x - 22, y: yInput - 21, width: 44, height: 42, rx: 8,
        class: inputValue === 1 ? "bptt-input-node bptt-input-cue" : "bptt-input-node",
      });
      appendSvg(unrollSvg, "text", { x, y: yInput - 3, class: "bptt-svg-value", "text-anchor": "middle" }, "x" + time);
      appendSvg(unrollSvg, "text", { x, y: yInput + 13, class: "bptt-svg-small", "text-anchor": "middle" }, inputValue === 1 ? "1.00" : "0.00");
      appendSvg(unrollSvg, "line", {
        x1: x, y1: yInput + 23, x2: x, y2: yState - 35, class: "bptt-input-edge",
        "marker-end": "url(#bptt-forward-arrow)",
      });

      if (time > 1) {
        const leftX = x - cellWidth + 36;
        const rightX = x - 36;
        appendSvg(unrollSvg, "line", {
          x1: leftX, y1: yState, x2: rightX, y2: yState, class: "bptt-forward-edge",
          "marker-end": "url(#bptt-forward-arrow)",
        });
        appendSvg(unrollSvg, "text", {
          x: (leftX + rightX) / 2, y: yState - 18, class: "bptt-edge-label", "text-anchor": "middle",
        }, "Wₕ");
        const strength = normalizedGradient(data.gradientHidden[time - 1]);
        appendSvg(unrollSvg, "line", {
          x1: rightX, y1: yBackward, x2: leftX, y2: yBackward, class: "bptt-backward-edge",
          "stroke-width": 1.8 + 6 * strength, "stroke-opacity": 0.28 + 0.72 * strength,
          "marker-end": "url(#bptt-backward-arrow)",
        });
      }

      const strength = normalizedGradient(data.gradientHidden[time]);
      appendSvg(unrollSvg, "rect", {
        x: x - 35, y: yState - 31, width: 70, height: 62, rx: 10, class: "bptt-state-node",
        fill: "rgba(207, 108, 55, " + (0.14 + 0.58 * strength).toFixed(3) + ")",
      });
      appendSvg(unrollSvg, "text", { x, y: yState - 9, class: "bptt-svg-value", "text-anchor": "middle" }, "h" + time);
      appendSvg(unrollSvg, "text", { x, y: yState + 9, class: "bptt-svg-vector", "text-anchor": "middle" }, formatVector(data.hidden[time], 2));
      appendSvg(unrollSvg, "text", { x, y: yBackward + 22, class: "bptt-backward-label", "text-anchor": "middle" }, "||g|| " + format(vectorNorm(data.gradientHidden[time])));
    }

    const outputX = 76 + (state.length - 1) * cellWidth + 78;
    appendSvg(unrollSvg, "line", {
      x1: outputX - 46, y1: yState, x2: outputX - 8, y2: yState, class: "bptt-forward-edge",
      "marker-end": "url(#bptt-forward-arrow)",
    });
    appendSvg(unrollSvg, "rect", {
      x: outputX, y: yState - 30, width: 98, height: 60, rx: 10, class: "bptt-loss-node",
    });
    appendSvg(unrollSvg, "text", { x: outputX + 49, y: yState - 7, class: "bptt-svg-value", "text-anchor": "middle" }, "loss L");
    appendSvg(unrollSvg, "text", { x: outputX + 49, y: yState + 14, class: "bptt-svg-small", "text-anchor": "middle" }, format(data.loss, 3));
    appendSvg(unrollSvg, "text", { x: outputX + 49, y: yState + 55, class: "bptt-output-note", "text-anchor": "middle" }, "ŷ = " + format(data.prediction, 2) + ", y = 1");
  }

  function drawMultiSeriesChart(svg, series, options) {
    clear(svg);
    const width = 500;
    const height = 230;
    const margins = { top: 18, right: 18, bottom: 43, left: 66 };
    const plotWidth = width - margins.left - margins.right;
    const plotHeight = height - margins.top - margins.bottom;
    const values = series.flatMap((item) => item.values);
    const yRange = options.yRange || (() => {
      const low = Math.min(...values);
      const high = Math.max(...values);
      const span = Math.max(0.15, high - low);
      return [low - 0.15 * span, high + 0.15 * span];
    })();
    const yLow = yRange[0];
    const yHigh = yRange[1];
    const length = series[0].values.length;
    const xScale = (index) => margins.left + (index / Math.max(length - 1, 1)) * plotWidth;
    const yScale = (value) => margins.top + (1 - (value - yLow) / Math.max(yHigh - yLow, 1e-9)) * plotHeight;

    appendSvg(svg, "line", { x1: margins.left, y1: margins.top, x2: margins.left, y2: margins.top + plotHeight, class: "bptt-chart-axis" });
    appendSvg(svg, "line", { x1: margins.left, y1: margins.top + plotHeight, x2: margins.left + plotWidth, y2: margins.top + plotHeight, class: "bptt-chart-axis" });
    for (let tick = 0; tick <= 4; tick += 1) {
      const ratio = tick / 4;
      const y = margins.top + (1 - ratio) * plotHeight;
      const value = yLow + ratio * (yHigh - yLow);
      appendSvg(svg, "line", { x1: margins.left, y1: y, x2: margins.left + plotWidth, y2: y, class: "bptt-chart-gridline" });
      appendSvg(svg, "text", { x: margins.left - 9, y: y + 4, class: "bptt-chart-tick", "text-anchor": "end" }, format(value, options.yDigits || 2));
    }
    const labelCount = Math.min(length, 5);
    for (let label = 0; label < labelCount; label += 1) {
      const index = Math.round(label * (length - 1) / Math.max(labelCount - 1, 1));
      appendSvg(svg, "text", { x: xScale(index), y: margins.top + plotHeight + 20, class: "bptt-chart-tick", "text-anchor": "middle" }, String(index + 1));
    }
    appendSvg(svg, "text", { x: margins.left + plotWidth / 2, y: height - 6, class: "bptt-chart-axis-label", "text-anchor": "middle" }, "time step");
    appendSvg(svg, "text", {
      x: 15, y: margins.top + plotHeight / 2, class: "bptt-chart-axis-label",
      transform: "rotate(-90 15 " + (margins.top + plotHeight / 2) + ")", "text-anchor": "middle",
    }, options.yLabel);

    series.forEach((item) => {
      const path = item.values.map((value, index) => (index === 0 ? "M" : "L") + xScale(index) + "," + yScale(value)).join(" ");
      appendSvg(svg, "path", { d: path, fill: "none", stroke: item.color, "stroke-linecap": "round", "stroke-linejoin": "round", "stroke-width": 2.7 });
      item.values.forEach((value, index) => appendSvg(svg, "circle", { cx: xScale(index), cy: yScale(value), r: 3.1, fill: item.color, stroke: "#fff", "stroke-width": 1.15 }));
    });
    series.forEach((item, index) => {
      const x = width - margins.right - 86;
      const y = margins.top + 13 + index * 17;
      appendSvg(svg, "line", { x1: x, y1: y, x2: x + 18, y2: y, stroke: item.color, "stroke-width": 3 });
      appendSvg(svg, "text", { x: x + 24, y: y + 4, class: "bptt-chart-tick" }, item.label);
    });
  }

  function renderMetrics(data) {
    const terminalNorm = Math.max(vectorNorm(data.gradientHidden[state.length]), 1e-12);
    const earlyNorm = vectorNorm(data.gradientHidden[1]);
    const ratio = earlyNorm / terminalNorm;
    if (ratio < 0.1) {
      summaryCopy.textContent = "The first hidden-state gradient has only " + (100 * ratio).toFixed(1) + "% of the terminal norm: it has vanished substantially.";
    } else if (ratio > 10) {
      summaryCopy.textContent = "The first hidden-state gradient has " + format(ratio, 1) + "× the terminal norm: it has been strongly amplified.";
    } else {
      summaryCopy.textContent = "The early and terminal hidden-state gradient norms have a similar scale (" + format(ratio, 2) + "×).";
    }
    const entries = [
      { label: "Final state", value: formatVector(data.hidden[state.length], 3), detail: "hT ∈ ℝ²" },
      { label: "Prediction and loss", value: "ŷ = " + format(data.prediction, 3), detail: "L = " + format(data.loss, 4) },
      { label: "Matrix-gradient norm", value: format(matrixNorm(data.gradientMatrix)), detail: "||∂L/∂Wₕ||F" },
      { label: "Early / terminal signal", value: format(ratio, 3) + "×", detail: "||∂L/∂h₁||₂ / ||∂L/∂hT||₂" },
    ];
    metrics.replaceChildren(...entries.map((entry) => {
      const card = document.createElement("div");
      card.className = "bptt-metric";
      const label = document.createElement("span");
      label.textContent = entry.label;
      const value = document.createElement("strong");
      value.textContent = entry.value;
      const detail = document.createElement("small");
      detail.textContent = entry.detail;
      card.append(label, value, detail);
      return card;
    }));
  }

  function renderTable(data) {
    const fragment = document.createDocumentFragment();
    for (let time = 1; time <= state.length; time += 1) {
      const row = document.createElement("tr");
      if (time === 1 || time === state.length) row.className = "bptt-table-emphasis";
      const values = [
        time,
        format(data.inputs[time], 2),
        formatVector(data.hidden[time - 1], 3),
        formatVector(data.hidden[time], 3),
        format(vectorNorm(data.delta[time])),
        format(matrixNorm(data.contributions[time])),
      ];
      values.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = String(value);
        row.append(cell);
      });
      fragment.append(row);
    }
    tableBody.replaceChildren(fragment);
  }

  function renderCharts(data) {
    drawMultiSeriesChart(stateChart, [
      { label: "h₁", values: data.hidden.slice(1).map((vector) => vector[0]), color: "#457eaa" },
      { label: "h₂", values: data.hidden.slice(1).map((vector) => vector[1]), color: "#77a86d" },
    ], { yLabel: "state value", yDigits: 2 });
    drawMultiSeriesChart(gradientChart, [
      { label: "||g||₂", values: data.gradientHidden.slice(1).map((vector) => logMagnitude(vectorNorm(vector))), color: "#cf6c37" },
    ], { yLabel: "log₁₀ ||gradient||₂", yRange: [-12, 5], yDigits: 0 });
  }

  function render() {
    setControls();
    const data = simulate();
    renderMatrices(data);
    drawUnroll(data);
    renderMetrics(data);
    renderCharts(data);
    renderTable(data);
  }

  function markCustom() {
    state.activePreset = "custom";
  }

  function setPreset(name) {
    Object.assign(state, PRESETS[name], { activePreset: name });
    status.textContent = PRESETS[name].status;
    render();
  }

  presetButtons.forEach((button) => {
    button.addEventListener("click", () => setPreset(button.dataset.bpttPreset));
  });
  activationInput.addEventListener("change", () => {
    state.activation = activationInput.value;
    markCustom();
    status.textContent = "Activation changed. Inspect the elementwise derivative in the backward path.";
    render();
  });
  weightInput.addEventListener("input", () => {
    state.scale = Number(weightInput.value);
    markCustom();
    status.textContent = "Matrix scale changed. The same Wₕ is used at every transition.";
    render();
  });
  lengthInput.addEventListener("input", () => {
    state.length = Number(lengthInput.value);
    markCustom();
    status.textContent = "Sequence length changed. More steps mean more matrix-Jacobian factors in BPTT.";
    render();
  });
  rateInput.addEventListener("input", () => {
    state.rate = Number(rateInput.value);
    status.textContent = "Learning rate changed.";
    render();
  });
  stepButton.addEventListener("click", () => {
    const data = simulate();
    const previousScale = state.scale;
    const proposed = previousScale - state.rate * data.gradientScale;
    if (proposed < SCALE_LIMITS.min || proposed > SCALE_LIMITS.max) {
      status.textContent = "The proposed α update is " + format(proposed, 3) + ", outside the explorer range. This is the practical danger of a large gradient; reduce η or reset.";
      return;
    }
    state.scale = proposed;
    markCustom();
    status.textContent = "Updated α from " + format(previousScale, 3) + " to " + format(state.scale, 3) + " using ⟨∂L/∂Wₕ, R⟩F.";
    render();
  });
  resetButton.addEventListener("click", () => setPreset("vanishing"));

  render();
})();
