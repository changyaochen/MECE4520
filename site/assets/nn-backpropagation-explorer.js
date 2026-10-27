(() => {
  const root = document.getElementById("nn-backprop-explorer");
  if (!root) return;

  const stepButton = root.querySelector("#nn-backprop-step");
  const runButton = root.querySelector("#nn-backprop-run");
  const resetButton = root.querySelector("#nn-backprop-reset");
  const status = root.querySelector("#nn-backprop-status");
  const rateInput = root.querySelector("#nn-backprop-rate");
  const rateValue = root.querySelector("#nn-backprop-rate-value");
  const datasetControls = root.querySelector("#nn-backprop-dataset-controls");
  const networkSvg = root.querySelector("#nn-backprop-network");
  const planeSvg = root.querySelector("#nn-backprop-plane");
  const timeline = root.querySelector("#nn-backprop-timeline");
  const trace = root.querySelector("#nn-backprop-trace");
  const traceDescription = root.querySelector("#nn-backprop-trace-description");
  const parameters = root.querySelector("#nn-backprop-parameters");
  const metrics = root.querySelector("#nn-backprop-metrics");
  const lossChart = root.querySelector("#nn-backprop-loss-chart");
  if (!stepButton || !runButton || !resetButton || !status || !rateInput || !rateValue
      || !datasetControls || !networkSvg || !planeSvg || !timeline || !trace
      || !traceDescription || !parameters || !metrics || !lossChart) return;

  const SVG_NS = "http://www.w3.org/2000/svg";
  const DOMAIN = [-2, 2];
  const RUN_UPDATES = 50;
  const DEFAULT_PARAMETERS = {
    weightsHidden: [[1.2, -0.8], [-1.0, 1.1], [0.4, -0.5]],
    biasHidden: [0.1, 0.25, 0.2],
    weightsOutput: [1.4, 1.1, -0.9],
    biasOutput: -0.6,
  };
  const STAGES = [
    { title: "Forward pass", description: "Compute the selected observation’s probability." },
    { title: "Loss and output error", description: "Compare that probability with its label." },
    { title: "Backpropagate", description: "Carry the output error through the hidden layer." },
    { title: "Full-dataset gradient", description: "Average gradients across all labeled observations." },
    { title: "Update", description: "Move every parameter opposite its gradient." },
  ];

  let parametersState = cloneParameters(DEFAULT_PARAMETERS);
  let learningRate = Number(rateInput.value);
  let selectedDataset = "moons";
  let selectedIndex = 18;
  let stage = 0;
  let updateCount = 0;
  let lastUpdate = null;
  let runToken = 0;
  let lossHistory = [];

  function makeMoonPoints() {
    const points = [];
    const count = 44;
    for (let index = 0; index < count; index += 1) {
      const fraction = index / (count - 1);
      const angle = Math.PI * fraction;
      const wobble = 0.055 * Math.sin(index * 2.3);
      points.push({
        x: -0.55 + Math.cos(angle),
        y: 0.35 + Math.sin(angle) + wobble,
        label: 1,
      });
      points.push({
        x: 0.55 - Math.cos(angle),
        y: -0.35 - Math.sin(angle) - wobble,
        label: 0,
      });
    }
    return points;
  }

  function makeSpiralPoints() {
    const points = [];
    const count = 54;
    for (let index = 0; index < count; index += 1) {
      const fraction = index / (count - 1);
      const radius = 0.12 + 1.48 * fraction;
      const angle = 1.95 * Math.PI * fraction;
      const wobble = 0.035 * Math.sin(index * 2.1);
      points.push({
        x: radius * Math.cos(angle) + wobble,
        y: radius * Math.sin(angle) - wobble,
        label: 1,
      });
      points.push({
        x: radius * Math.cos(angle + Math.PI) - wobble,
        y: radius * Math.sin(angle + Math.PI) + wobble,
        label: 0,
      });
    }
    return points;
  }

  const DATASETS = {
    moons: { label: "Half moons", points: makeMoonPoints() },
    spirals: { label: "Intertwining spirals", points: makeSpiralPoints() },
  };

  function cloneParameters(source) {
    return {
      weightsHidden: source.weightsHidden.map((row) => [...row]),
      biasHidden: [...source.biasHidden],
      weightsOutput: [...source.weightsOutput],
      biasOutput: source.biasOutput,
    };
  }

  function zeroGradients() {
    return {
      weightsHidden: parametersState.weightsHidden.map((row) => row.map(() => 0)),
      biasHidden: parametersState.biasHidden.map(() => 0),
      weightsOutput: parametersState.weightsOutput.map(() => 0),
      biasOutput: 0,
    };
  }

  function sigmoid(value) {
    const bounded = Math.max(-40, Math.min(40, value));
    return 1 / (1 + Math.exp(-bounded));
  }

  function relu(value) {
    return Math.max(0, value);
  }

  function reluDerivative(value) {
    return value > 0 ? 1 : 0;
  }

  function binaryCrossEntropy(probability, label) {
    const bounded = Math.max(1e-8, Math.min(1 - 1e-8, probability));
    return -(label * Math.log(bounded) + (1 - label) * Math.log(1 - bounded));
  }

  function forward(input) {
    const hiddenContributions = parametersState.weightsHidden.map((row) => (
      row.map((weight, index) => weight * input[index])
    ));
    const hiddenScores = hiddenContributions.map((contributions, index) => (
      contributions.reduce((sum, contribution) => sum + contribution, parametersState.biasHidden[index])
    ));
    const hiddenActivations = hiddenScores.map(relu);
    const outputContributions = hiddenActivations.map((activation, index) => (
      activation * parametersState.weightsOutput[index]
    ));
    const outputScore = outputContributions.reduce(
      (sum, contribution) => sum + contribution,
      parametersState.biasOutput,
    );
    return {
      input,
      hiddenContributions,
      hiddenScores,
      hiddenActivations,
      outputContributions,
      outputScore,
      probability: sigmoid(outputScore),
    };
  }

  function gradientsForObservation(point) {
    const output = forward([point.x, point.y]);
    const outputError = output.probability - point.label;
    const hiddenErrors = output.hiddenScores.map((score, index) => (
      outputError * parametersState.weightsOutput[index] * reluDerivative(score)
    ));
    return {
      output,
      loss: binaryCrossEntropy(output.probability, point.label),
      outputError,
      hiddenErrors,
      gradients: {
        weightsHidden: hiddenErrors.map((error) => [error * point.x, error * point.y]),
        biasHidden: [...hiddenErrors],
        weightsOutput: output.hiddenActivations.map((activation) => outputError * activation),
        biasOutput: outputError,
      },
    };
  }

  function evaluateDataset() {
    const points = DATASETS[selectedDataset].points;
    const gradients = zeroGradients();
    let loss = 0;
    let correct = 0;
    const probabilities = [];
    points.forEach((point) => {
      const observation = gradientsForObservation(point);
      loss += observation.loss;
      probabilities.push(observation.output.probability);
      correct += Number((observation.output.probability >= 0.5 ? 1 : 0) === point.label);
      observation.gradients.weightsHidden.forEach((row, hiddenIndex) => {
        row.forEach((value, inputIndex) => {
          gradients.weightsHidden[hiddenIndex][inputIndex] += value;
        });
      });
      observation.gradients.biasHidden.forEach((value, hiddenIndex) => {
        gradients.biasHidden[hiddenIndex] += value;
      });
      observation.gradients.weightsOutput.forEach((value, hiddenIndex) => {
        gradients.weightsOutput[hiddenIndex] += value;
      });
      gradients.biasOutput += observation.gradients.biasOutput;
    });
    const count = points.length;
    gradients.weightsHidden = gradients.weightsHidden.map((row) => row.map((value) => value / count));
    gradients.biasHidden = gradients.biasHidden.map((value) => value / count);
    gradients.weightsOutput = gradients.weightsOutput.map((value) => value / count);
    gradients.biasOutput /= count;
    return { loss: loss / count, accuracy: correct / count, gradients, probabilities };
  }

  function applyGradient(gradients) {
    parametersState.weightsHidden.forEach((row, hiddenIndex) => {
      row.forEach((_weight, inputIndex) => {
        parametersState.weightsHidden[hiddenIndex][inputIndex] -= learningRate
          * gradients.weightsHidden[hiddenIndex][inputIndex];
      });
    });
    parametersState.biasHidden.forEach((_bias, hiddenIndex) => {
      parametersState.biasHidden[hiddenIndex] -= learningRate * gradients.biasHidden[hiddenIndex];
    });
    parametersState.weightsOutput.forEach((_weight, hiddenIndex) => {
      parametersState.weightsOutput[hiddenIndex] -= learningRate * gradients.weightsOutput[hiddenIndex];
    });
    parametersState.biasOutput -= learningRate * gradients.biasOutput;
  }

  function performTrainingUpdate() {
    const before = evaluateDataset();
    applyGradient(before.gradients);
    const after = evaluateDataset();
    updateCount += 1;
    lastUpdate = {
      gradients: before.gradients,
      lossBefore: before.loss,
      lossAfter: after.loss,
      accuracyBefore: before.accuracy,
      accuracyAfter: after.accuracy,
    };
    lossHistory.push(after.loss);
    return after;
  }

  function resetLossHistory() {
    lossHistory = [evaluateDataset().loss];
  }

  function format(value, digits = 3) {
    const rounded = Number(value).toFixed(digits);
    return rounded === "-0.000" ? "0.000" : rounded;
  }

  function percentage(value) {
    return `${(100 * value).toFixed(1)}%`;
  }

  function createElement(name, className, text) {
    const element = document.createElement(name);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function svgElement(name, attributes = {}) {
    const element = document.createElementNS(SVG_NS, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
  }

  function addSvgText(svg, text, x, y, attributes = {}) {
    const element = svgElement("text", { x, y, ...attributes });
    element.textContent = text;
    svg.append(element);
    return element;
  }

  function interpolate(start, end, amount) {
    return Math.round(start + (end - start) * amount);
  }

  function probabilityColor(probability) {
    const low = [191, 81, 77];
    const middle = [246, 246, 239];
    const high = [59, 137, 99];
    const scaled = probability <= 0.5 ? probability * 2 : (probability - 0.5) * 2;
    const from = probability <= 0.5 ? low : middle;
    const to = probability <= 0.5 ? middle : high;
    return `rgb(${interpolate(from[0], to[0], scaled)}, ${interpolate(from[1], to[1], scaled)}, ${interpolate(from[2], to[2], scaled)})`;
  }

  function planeCoordinate(value, start, size) {
    return start + (value - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0]) * size;
  }

  function selectedPoint() {
    return DATASETS[selectedDataset].points[selectedIndex];
  }

  function selectedObservation() {
    return gradientsForObservation(selectedPoint());
  }

  function renderDatasetControls() {
    datasetControls.replaceChildren();
    Object.entries(DATASETS).forEach(([key, dataset]) => {
      const button = createElement(
        "button",
        `nn-dataset-button${selectedDataset === key ? " nn-dataset-button-active" : ""}`,
        dataset.label,
      );
      button.type = "button";
      button.setAttribute("aria-pressed", String(selectedDataset === key));
      button.addEventListener("click", () => {
        selectedDataset = key;
        selectedIndex = Math.min(18, DATASETS[key].points.length - 1);
        parametersState = cloneParameters(DEFAULT_PARAMETERS);
        updateCount = 0;
        lastUpdate = null;
        stage = 0;
        resetLossHistory();
        renderAll();
        status.textContent = `${dataset.label} selected and model reset`;
      });
      datasetControls.append(button);
    });
  }

  function renderTimeline() {
    timeline.replaceChildren();
    STAGES.forEach((item, index) => {
      const entry = createElement(
        "li",
        `nn-backprop-timeline-item${stage === index + 1 ? " nn-backprop-timeline-active" : ""}${stage > index + 1 ? " nn-backprop-timeline-complete" : ""}`,
      );
      const number = createElement("span", "nn-backprop-stage-number", String(index + 1));
      const copy = createElement("div", "nn-backprop-stage-copy");
      copy.append(
        createElement("strong", null, item.title),
        createElement("span", null, item.description),
      );
      entry.append(number, copy);
      timeline.append(entry);
    });
  }

  function renderPlane() {
    planeSvg.replaceChildren();
    planeSvg.setAttribute("viewBox", "0 0 420 380");
    const defs = svgElement("defs");
    const marker = svgElement("marker", {
      id: "nn-selected-point-arrow", markerWidth: 8, markerHeight: 8, refX: 7, refY: 4,
      orient: "auto", markerUnits: "strokeWidth",
    });
    marker.append(svgElement("path", { d: "M 0 0 L 8 4 L 0 8 z", class: "nn-selected-point-arrowhead" }));
    defs.append(marker);
    planeSvg.append(defs);
    const left = 50;
    const top = 22;
    const width = 340;
    const height = 315;
    const resolution = 44;
    const cellWidth = width / resolution;
    const cellHeight = height / resolution;
    const grid = svgElement("g", { "aria-hidden": "true" });
    for (let row = 0; row < resolution; row += 1) {
      for (let column = 0; column < resolution; column += 1) {
        const x1 = DOMAIN[0] + (column + 0.5) / resolution * (DOMAIN[1] - DOMAIN[0]);
        const x2 = DOMAIN[1] - (row + 0.5) / resolution * (DOMAIN[1] - DOMAIN[0]);
        const cell = svgElement("rect", {
          x: left + column * cellWidth,
          y: top + row * cellHeight,
          width: cellWidth + 0.3,
          height: cellHeight + 0.3,
          fill: probabilityColor(forward([x1, x2]).probability),
        });
        grid.append(cell);
      }
    }
    planeSvg.append(grid);
    planeSvg.append(
      svgElement("line", { x1: left, y1: top + height, x2: left + width, y2: top + height, class: "nn-axis-line" }),
      svgElement("line", { x1: left, y1: top, x2: left, y2: top + height, class: "nn-axis-line" }),
    );
    [-2, -1, 0, 1, 2].forEach((tick) => {
      const x = planeCoordinate(tick, left, width);
      const y = top + height - (tick - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0]) * height;
      planeSvg.append(
        svgElement("line", { x1: x, y1: top, x2: x, y2: top + height, class: "nn-grid-line" }),
        svgElement("line", { x1: left, y1: y, x2: left + width, y2: y, class: "nn-grid-line" }),
      );
      addSvgText(planeSvg, String(tick), x, top + height + 20, {
        class: "nn-axis-text", "text-anchor": "middle",
      });
      addSvgText(planeSvg, String(tick), left - 10, y + 4, {
        class: "nn-axis-text", "text-anchor": "end",
      });
    });
    addSvgText(planeSvg, "x₁", left + width / 2, 372, {
      class: "nn-axis-label", "text-anchor": "middle",
    });
    addSvgText(planeSvg, "x₂", 15, top + height / 2, {
      class: "nn-axis-label", transform: `rotate(-90 15 ${top + height / 2})`, "text-anchor": "middle",
    });
    const points = DATASETS[selectedDataset].points;
    points.forEach((point, index) => {
      const circle = svgElement("circle", {
        cx: planeCoordinate(point.x, left, width),
        cy: top + height - (point.y - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0]) * height,
        r: index === selectedIndex ? 6.4 : 4.2,
        class: `nn-data-point nn-data-point-${point.label === 1 ? "positive" : "negative"}${index === selectedIndex ? " nn-backprop-selected-point" : ""}`,
        tabindex: 0,
        role: "button",
        "aria-label": `Select observation ${index + 1}, label ${point.label}`,
      });
      circle.addEventListener("click", () => {
        selectedIndex = index;
        stage = 0;
        renderAll();
        status.textContent = `Selected observation ${index + 1}, label ${point.label}`;
      });
      circle.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          circle.dispatchEvent(new Event("click"));
        }
      });
      const title = svgElement("title");
      title.textContent = `Observation ${index + 1}: x₁ = ${format(point.x, 2)}, x₂ = ${format(point.y, 2)}, label = ${point.label}`;
      circle.append(title);
      planeSvg.append(circle);
    });
    const point = selectedPoint();
    const markerX = planeCoordinate(point.x, left, width);
    const markerY = top + height - (point.y - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0]) * height;
    const labelOnLeft = markerX > left + width - 72;
    const labelX = markerX + (labelOnLeft ? -44 : 44);
    const labelY = Math.max(top + 17, markerY - 38);
    planeSvg.append(svgElement("path", {
      d: `M ${labelX} ${labelY + 5} L ${markerX} ${markerY - 4}`,
      fill: "none",
      class: "nn-selected-point-arrow",
      "marker-end": "url(#nn-selected-point-arrow)",
    }));
    addSvgText(planeSvg, "selected", labelX, labelY, {
      class: "nn-marker-label",
      "text-anchor": "middle",
    });
  }

  function curvedEdgePath(from, to, curveDirection = 1) {
    const horizontalControlOffset = (to.x - from.x) * 0.4;
    const isHorizontal = Math.abs(to.y - from.y) < 0.1;
    const bow = isHorizontal ? 30 * curveDirection : 0;
    return [
      `M ${from.x} ${from.y}`,
      `C ${from.x + horizontalControlOffset} ${from.y + bow}`,
      `${to.x - horizontalControlOffset} ${to.y + bow}`,
      `${to.x} ${to.y}`,
    ].join(" ");
  }

  function weightWidth(weight) {
    return Math.min(14, 2 + 7 * Math.abs(weight));
  }

  function drawEdge(svg, from, to, weight, colorClass, active, curveDirection) {
    svg.append(svgElement("path", {
      d: curvedEdgePath(from, to, curveDirection),
      fill: "none",
      "stroke-width": weightWidth(weight),
      class: `nn-network-edge ${colorClass}${active ? " nn-network-edge-active" : ""}`,
    }));
  }

  function drawBackpropEdge(svg, from, to, curveDirection) {
    svg.append(svgElement("path", {
      d: curvedEdgePath(from, to, curveDirection),
      fill: "none",
      class: "nn-backprop-signal",
      "marker-end": "url(#nn-backprop-arrow)",
    }));
  }

  function appendNodeNotation(text, notation) {
    const base = svgElement("tspan");
    base.textContent = notation.base;
    text.append(base);
    if (notation.subscript) {
      const subscript = svgElement("tspan", {
        "baseline-shift": "sub",
        "font-size": "0.72em",
      });
      subscript.textContent = notation.subscript;
      text.append(subscript);
    }
    if (notation.superscript) {
      const superscript = svgElement("tspan", {
        "baseline-shift": "super",
        "font-size": "0.68em",
      });
      superscript.textContent = notation.superscript;
      text.append(superscript);
    }
    if (notation.suffix) {
      const suffix = svgElement("tspan", { "baseline-shift": "baseline" });
      suffix.textContent = notation.suffix;
      text.append(suffix);
    }
  }

  function drawNode(svg, position, lines, nodeClass, active, width = 124) {
    const height = 23 + lines.length * 17;
    const group = svgElement("g", { class: `nn-network-node ${nodeClass}${active ? " nn-network-node-active" : ""}` });
    group.append(svgElement("rect", {
      x: position.x - width / 2, y: position.y - height / 2, width, height, rx: 9,
    }));
    lines.forEach((line, index) => {
      const text = svgElement("text", {
        x: position.x,
        y: position.y - (lines.length - 1) * 8.5 + index * 17 + 4,
        "text-anchor": "middle",
        class: index === 0 ? "nn-node-title" : "nn-node-value",
      });
      if (typeof line === "string") {
        text.textContent = line;
      } else {
        appendNodeNotation(text, line);
      }
      group.append(text);
    });
    svg.append(group);
  }

  function appendMathSegments(element, segments) {
    segments.forEach((segment) => {
      if (typeof segment === "string") {
        element.append(document.createTextNode(segment));
        return;
      }
      const variable = document.createElement("i");
      variable.textContent = segment.base;
      const superscript = document.createElement("sup");
      superscript.textContent = segment.superscript;
      element.append(variable, superscript);
    });
  }

  function renderNetwork() {
    const observation = selectedObservation();
    const output = observation.output;
    networkSvg.replaceChildren();
    networkSvg.setAttribute("viewBox", "0 0 750 450");
    const defs = svgElement("defs");
    const marker = svgElement("marker", {
      id: "nn-backprop-arrow", markerWidth: 8, markerHeight: 8, refX: 7, refY: 4,
      orient: "auto", markerUnits: "strokeWidth",
    });
    marker.append(svgElement("path", { d: "M 0 0 L 8 4 L 0 8 z", class: "nn-backprop-arrowhead" }));
    defs.append(marker);
    networkSvg.append(defs);
    const positions = {
      inputs: [{ x: 100, y: 120 }, { x: 100, y: 330 }],
      hidden: [{ x: 380, y: 80 }, { x: 380, y: 225 }, { x: 380, y: 370 }],
      output: { x: 650, y: 225 },
    };
    addSvgText(networkSvg, "input", 100, 28, { class: "nn-layer-title", "text-anchor": "middle" });
    addSvgText(networkSvg, "hidden layer (ReLU)", 380, 28, { class: "nn-layer-title", "text-anchor": "middle" });
    addSvgText(networkSvg, "output (sigmoid)", 650, 28, { class: "nn-layer-title", "text-anchor": "middle" });
    parametersState.weightsHidden.forEach((row, hiddenIndex) => {
      row.forEach((weight, inputIndex) => {
        drawEdge(
          networkSvg,
          positions.inputs[inputIndex],
          positions.hidden[hiddenIndex],
          weight,
          `nn-network-edge-hidden-${hiddenIndex + 1}`,
          stage === 1,
          hiddenIndex === 1 ? -1 : 1,
        );
      });
    });
    parametersState.weightsOutput.forEach((weight, hiddenIndex) => {
      drawEdge(
        networkSvg,
        positions.hidden[hiddenIndex],
        positions.output,
        weight,
        "nn-network-edge-output",
        stage === 1 || stage === 2,
        hiddenIndex === 1 ? -1 : 1,
      );
    });
    if (stage >= 3) {
      parametersState.weightsOutput.forEach((_weight, hiddenIndex) => {
        drawBackpropEdge(networkSvg, positions.output, positions.hidden[hiddenIndex], hiddenIndex === 1 ? -1 : 1);
      });
      parametersState.weightsHidden.forEach((_row, hiddenIndex) => {
        positions.inputs.forEach((inputPosition) => {
          drawBackpropEdge(networkSvg, positions.hidden[hiddenIndex], inputPosition, hiddenIndex === 1 ? -1 : 1);
        });
      });
    }
    positions.inputs.forEach((position, index) => {
      drawNode(
        networkSvg,
        position,
        [{ base: "x", subscript: String(index + 1) }, format(output.input[index], 2)],
        "nn-input-node",
        stage === 1,
      );
    });
    positions.hidden.forEach((position, index) => {
      drawNode(
        networkSvg,
        position,
        [
          { base: "z", subscript: String(index + 1), superscript: "[1]", suffix: ` = ${format(output.hiddenScores[index], 2)}` },
          { base: "a", subscript: String(index + 1), superscript: "[1]", suffix: ` = ${format(output.hiddenActivations[index], 2)}` },
        ],
        "nn-hidden-node",
        stage === 1 || stage === 3,
      );
    });
    drawNode(
      networkSvg,
      positions.output,
      [
        { base: "z", superscript: "[2]", suffix: ` = ${format(output.outputScore, 2)}` },
        { base: "a", superscript: "[2]", suffix: ` = ${format(output.probability)}` },
      ],
      output.probability >= 0.5 ? "nn-output-positive" : "nn-output-negative",
      stage === 2 || stage === 3,
    );
  }

  function metricItem(label, value, detail, tone = "neutral") {
    const item = createElement("div", `nn-backprop-metric nn-backprop-metric-${tone}`);
    const labelElement = createElement("span", "nn-backprop-metric-label");
    const detailElement = createElement("span", "nn-backprop-metric-detail");
    if (Array.isArray(label)) appendMathSegments(labelElement, label);
    else labelElement.textContent = label;
    if (Array.isArray(detail)) appendMathSegments(detailElement, detail);
    else detailElement.textContent = detail;
    item.append(
      labelElement,
      createElement("strong", null, value),
      detailElement,
    );
    return item;
  }

  function renderTrace() {
    const point = selectedPoint();
    const observation = selectedObservation();
    const hiddenErrors = observation.hiddenErrors.map((value) => format(value)).join(", ");
    traceDescription.textContent = `Observation ${selectedIndex + 1}: label ${point.label}. This trace is illustrative; the update averages all observations.`;
    trace.replaceChildren(
      metricItem("Input", `(${format(point.x, 2)}, ${format(point.y, 2)})`, `label y = ${point.label}`),
      metricItem(
        "Prediction",
        format(observation.output.probability),
        [{ base: "z", superscript: "[2]" }, ` = ${format(observation.output.outputScore)}`],
        observation.output.probability >= 0.5 ? "positive" : "negative",
      ),
      metricItem("Loss", format(observation.loss), "binary cross-entropy"),
      metricItem(
        ["Output error ", { base: "δ", superscript: "[2]" }],
        format(observation.outputError),
        [{ base: "δ", superscript: "[2]" }, " = ", { base: "a", superscript: "[2]" }, " − y"],
        observation.outputError >= 0 ? "negative" : "positive",
      ),
      metricItem(
        ["Hidden errors ", { base: "δ", superscript: "[1]" }],
        `[${hiddenErrors}]`,
        [
          { base: "δ", superscript: "[1]" }, " = (",
          { base: "δ", superscript: "[2]" }, { base: "W", superscript: "[2]" },
          ") ⊙ ", { base: "g", superscript: "[1]′" }, "(",
          { base: "z", superscript: "[1]" }, ")",
        ],
      ),
    );
  }

  function matrixSymbol(base, layer) {
    const symbol = createElement("span", "nn-backprop-matrix-symbol");
    const letter = createElement("i", null, base);
    const superscript = createElement("sup", null, `[${layer}]`);
    symbol.append(letter, superscript);
    return symbol;
  }

  function gradientValue(gradient) {
    return lastUpdate ? -learningRate * gradient : 0;
  }

  function parameterCell(value, gradient, rowClass) {
    const delta = gradientValue(gradient);
    const gradientReady = stage >= 4 || lastUpdate;
    const direction = lastUpdate
      ? (delta > 0 ? " nn-backprop-update-up" : (delta < 0 ? " nn-backprop-update-down" : ""))
      : "";
    const cell = createElement("span", `nn-backprop-parameter-cell ${rowClass}${direction}`);
    const detail = lastUpdate
      ? `Δ ${delta >= 0 ? "+" : ""}${format(delta, 3)}`
      : (gradientReady ? `g ${format(gradient, 3)}` : "awaiting gradient");
    cell.append(
      createElement("strong", null, format(value, 2)),
      createElement("small", null, detail),
    );
    return cell;
  }

  function parameterMatrix(base, layer, rows, columns, values, gradients, rowClass) {
    const block = createElement("div", "nn-backprop-matrix-block");
    const header = createElement("div", "nn-backprop-matrix-header");
    header.append(matrixSymbol(base, layer));
    const grid = createElement("div", "nn-backprop-matrix-grid");
    grid.style.gridTemplateColumns = `repeat(${columns}, minmax(0, 1fr))`;
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        const value = rows === 1 && columns === 1 ? values : values[row][column];
        const gradient = rows === 1 && columns === 1 ? gradients : gradients[row][column];
        const colorClass = typeof rowClass === "function" ? rowClass(row) : rowClass;
        grid.append(parameterCell(value, gradient, colorClass));
      }
    }
    block.append(header, grid);
    return block;
  }

  function renderParameters() {
    const evaluation = evaluateDataset();
    const displayedGradients = lastUpdate ? lastUpdate.gradients : evaluation.gradients;
    parameters.replaceChildren(
      parameterMatrix(
        "W", 1, 3, 2, parametersState.weightsHidden, displayedGradients.weightsHidden,
        (row) => `nn-backprop-row-${row + 1}`,
      ),
      parameterMatrix(
        "B", 1, 3, 1,
        parametersState.biasHidden.map((value) => [value]),
        displayedGradients.biasHidden.map((value) => [value]),
        (row) => `nn-backprop-row-${row + 1}`,
      ),
      parameterMatrix(
        "W", 2, 1, 3, [parametersState.weightsOutput], [displayedGradients.weightsOutput],
        "nn-backprop-row-output",
      ),
      parameterMatrix("B", 2, 1, 1, parametersState.biasOutput, displayedGradients.biasOutput, "nn-backprop-row-output"),
    );
  }

  function renderMetrics() {
    const evaluation = evaluateDataset();
    const lossDetail = lastUpdate
      ? `${format(lastUpdate.lossBefore)} → ${format(lastUpdate.lossAfter)} in last update`
      : "mean binary cross-entropy";
    const accuracyDetail = lastUpdate
      ? `${percentage(lastUpdate.accuracyBefore)} → ${percentage(lastUpdate.accuracyAfter)} in last update`
      : "threshold = 0.5";
    metrics.replaceChildren(
      metricItem("Training loss", format(evaluation.loss), lossDetail, "loss"),
      metricItem("Training accuracy", percentage(evaluation.accuracy), accuracyDetail, "positive"),
      metricItem("Updates completed", String(updateCount), `learning rate η = ${learningRate.toFixed(2)}`),
      metricItem("Mean-gradient update", lastUpdate ? "shown above" : "not yet applied", "green Δ increases a parameter; red Δ decreases it"),
    );
  }

  function renderLossChart() {
    const width = 520;
    const height = 190;
    const margin = { top: 16, right: 18, bottom: 33, left: 50 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const currentStep = lossHistory.length - 1;
    const maximumLoss = Math.max(0.05, ...lossHistory) * 1.06;
    const xPosition = (step) => margin.left + step / RUN_UPDATES * plotWidth;
    const yPosition = (loss) => margin.top + (1 - loss / maximumLoss) * plotHeight;

    lossChart.replaceChildren();
    lossChart.append(
      svgElement("line", {
        x1: margin.left, y1: margin.top + plotHeight,
        x2: width - margin.right, y2: margin.top + plotHeight,
        class: "nn-backprop-chart-axis",
      }),
      svgElement("line", {
        x1: margin.left, y1: margin.top,
        x2: margin.left, y2: margin.top + plotHeight,
        class: "nn-backprop-chart-axis",
      }),
    );

    [0, 0.5, 1].forEach((fraction) => {
      const y = margin.top + fraction * plotHeight;
      const value = maximumLoss * (1 - fraction);
      lossChart.append(
        svgElement("line", {
          x1: margin.left, y1: y, x2: width - margin.right, y2: y,
          class: "nn-backprop-chart-gridline",
        }),
      );
      addSvgText(lossChart, format(value, 2), margin.left - 8, y + 3.5, {
        class: "nn-backprop-chart-tick", "text-anchor": "end",
      });
    });

    [0, RUN_UPDATES / 2, RUN_UPDATES].forEach((step) => {
      const x = xPosition(step);
      lossChart.append(svgElement("line", {
        x1: x, y1: margin.top + plotHeight,
        x2: x, y2: margin.top + plotHeight + 4,
        class: "nn-backprop-chart-axis",
      }));
      addSvgText(lossChart, String(step), x, height - 12, {
        class: "nn-backprop-chart-tick", "text-anchor": "middle",
      });
    });

    if (lossHistory.length > 1) {
      const path = lossHistory.map((loss, index) => (
        `${index === 0 ? "M" : "L"}${xPosition(index).toFixed(2)},${yPosition(loss).toFixed(2)}`
      )).join(" ");
      lossChart.append(svgElement("path", { d: path, class: "nn-backprop-loss-line" }));
    }

    lossHistory.forEach((loss, index) => {
      lossChart.append(svgElement("circle", {
        cx: xPosition(index), cy: yPosition(loss),
        r: index === currentStep ? 4.1 : 2.3,
        class: index === currentStep ? "nn-backprop-loss-point-current" : "nn-backprop-loss-point",
      }));
    });

    addSvgText(lossChart, "update", margin.left + plotWidth / 2, height - 1, {
      class: "nn-backprop-chart-axis-label", "text-anchor": "middle",
    });
    addSvgText(lossChart, "loss", 13, margin.top + plotHeight / 2, {
      class: "nn-backprop-chart-axis-label",
      transform: `rotate(-90 13 ${margin.top + plotHeight / 2})`,
      "text-anchor": "middle",
    });
  }

  function renderAll() {
    rateValue.textContent = learningRate.toFixed(2);
    renderDatasetControls();
    renderTimeline();
    renderPlane();
    renderNetwork();
    renderTrace();
    renderParameters();
    renderMetrics();
    renderLossChart();
  }

  function stepThroughUpdate() {
    runToken += 1;
    if (stage < STAGES.length - 1) {
      stage += 1;
      status.textContent = `${stage} of 5: ${STAGES[stage - 1].description}`;
    } else if (stage === STAGES.length - 1) {
      performTrainingUpdate();
      stage = STAGES.length;
      status.textContent = `Update ${updateCount} applied using the mean full-dataset gradient`;
    } else {
      stage = 0;
      status.textContent = "Ready to trace the next update";
    }
    renderAll();
  }

  function runUpdates() {
    const token = runToken + 1;
    runToken = token;
    runButton.disabled = true;
    stepButton.disabled = true;
    let completed = 0;
    const advance = () => {
      if (token !== runToken) return;
      performTrainingUpdate();
      stage = STAGES.length;
      completed += 1;
      status.textContent = `Running update ${completed} of ${RUN_UPDATES}`;
      renderAll();
      if (completed < RUN_UPDATES) {
        window.setTimeout(advance, 85);
      } else {
        runButton.disabled = false;
        stepButton.disabled = false;
        status.textContent = `${RUN_UPDATES} full-dataset updates complete`;
      }
    };
    advance();
  }

  stepButton.addEventListener("click", stepThroughUpdate);
  runButton.addEventListener("click", runUpdates);
  resetButton.addEventListener("click", () => {
    runToken += 1;
    parametersState = cloneParameters(DEFAULT_PARAMETERS);
    learningRate = Number(rateInput.value);
    selectedDataset = "moons";
    selectedIndex = 18;
    updateCount = 0;
    lastUpdate = null;
    stage = 0;
    resetLossHistory();
    runButton.disabled = false;
    stepButton.disabled = false;
    renderAll();
    status.textContent = "Example restored";
  });
  rateInput.addEventListener("input", () => {
    learningRate = Number(rateInput.value);
    rateValue.textContent = learningRate.toFixed(2);
    renderMetrics();
  });

  resetLossHistory();
  renderAll();
})();
