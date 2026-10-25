(() => {
  const root = document.getElementById("nn-forward-explorer");
  if (!root) return;

  const inputControls = root.querySelector("#nn-input-controls");
  const datasetControls = root.querySelector("#nn-dataset-controls");
  const parameterControls = root.querySelector("#nn-parameter-controls");
  const probabilityPlane = root.querySelector("#nn-probability-plane");
  const networkDiagram = root.querySelector("#nn-network-diagram");
  const status = root.querySelector("#nn-status");
  const animateButton = root.querySelector("#nn-animate");
  const resetButton = root.querySelector("#nn-reset");
  if (!inputControls || !datasetControls || !parameterControls || !probabilityPlane || !networkDiagram
      || !status || !animateButton || !resetButton) return;

  const SVG_NS = "http://www.w3.org/2000/svg";
  const DOMAIN = [-2, 2];
  const PARAMETER_LIMIT = 3;
  const DEFAULT_STATE = {
    input: [0.8, -1.0],
    weightsHidden: [[1.2, -0.8], [-1.0, 1.1], [0.4, -0.5]],
    biasHidden: [0.1, 0.25, 0.2],
    weightsOutput: [1.4, 1.1, -0.9],
    biasOutput: -0.6,
  };
  let state = cloneState(DEFAULT_STATE);
  let selectedDataset = "moons";
  let animationPhase = 0;
  let animationTimer = null;

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

  function cloneState(source) {
    return {
      input: [...source.input],
      weightsHidden: source.weightsHidden.map((row) => [...row]),
      biasHidden: [...source.biasHidden],
      weightsOutput: [...source.weightsOutput],
      biasOutput: source.biasOutput,
    };
  }

  function sigmoid(value) {
    const bounded = Math.max(-40, Math.min(40, value));
    return 1 / (1 + Math.exp(-bounded));
  }

  function relu(value) {
    return Math.max(0, value);
  }

  function weightWidth(weight) {
    return Math.min(14, 2 + 7 * Math.abs(weight));
  }

  function forward(input) {
    const hiddenContributions = state.weightsHidden.map((row) => (
      row.map((weight, index) => weight * input[index])
    ));
    const hiddenScores = hiddenContributions.map((contributions, index) => (
      contributions.reduce((sum, contribution) => sum + contribution, state.biasHidden[index])
    ));
    const hiddenActivations = hiddenScores.map(relu);
    const outputContributions = hiddenActivations.map((activation, index) => (
      activation * state.weightsOutput[index]
    ));
    const outputScore = outputContributions.reduce(
      (sum, contribution) => sum + contribution,
      state.biasOutput,
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

  function createElement(name, className) {
    const element = document.createElement(name);
    if (className) element.className = className;
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

  function clampInput(value) {
    return Math.max(DOMAIN[0], Math.min(DOMAIN[1], value));
  }

  function clampParameter(value) {
    return Math.max(-PARAMETER_LIMIT, Math.min(PARAMETER_LIMIT, value));
  }

  function format(value, digits = 2) {
    return Number(value).toFixed(digits);
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

  function updateAfterChange() {
    stopAnimation();
    renderViews();
  }

  function appendHtmlInputNotation(element, subscript) {
    const base = document.createElement("i");
    base.textContent = "x";
    const index = document.createElement("sub");
    index.textContent = subscript;
    element.append(base, index);
  }

  function makeInputControl(index) {
    const card = createElement("div", "nn-input-card");
    const label = createElement("label");
    label.htmlFor = `nn-input-${index}`;
    appendHtmlInputNotation(label, String(index + 1));
    const number = createElement("input", "nn-number-input");
    number.id = `nn-input-${index}`;
    number.type = "number";
    number.min = String(DOMAIN[0]);
    number.max = String(DOMAIN[1]);
    number.step = "0.05";
    number.value = format(state.input[index]);
    number.setAttribute("aria-label", `Numeric value of input x subscript ${index + 1}`);
    const slider = createElement("input", "nn-input-slider");
    slider.type = "range";
    slider.min = String(DOMAIN[0]);
    slider.max = String(DOMAIN[1]);
    slider.step = "0.05";
    slider.value = String(state.input[index]);
    slider.setAttribute("aria-label", `Slider for input x subscript ${index + 1}`);
    const setInput = (value) => {
      const parsed = Number.parseFloat(value);
      if (!Number.isFinite(parsed)) return;
      state.input[index] = clampInput(parsed);
      number.value = format(state.input[index]);
      slider.value = String(state.input[index]);
      updateAfterChange();
    };
    number.addEventListener("change", () => setInput(number.value));
    slider.addEventListener("input", () => setInput(slider.value));
    card.append(label, number, slider);
    return card;
  }

  function parameterInput({ label, value, update, rowClass }) {
    const input = createElement(
      "input",
      `nn-number-input nn-parameter-input ${rowClass}`,
    );
    input.type = "number";
    input.min = String(-PARAMETER_LIMIT);
    input.max = String(PARAMETER_LIMIT);
    input.step = "0.1";
    input.value = format(value);
    input.setAttribute("aria-label", label);
    const commitParameter = (rawValue) => {
      const parsed = Number.parseFloat(rawValue);
      if (!Number.isFinite(parsed)) {
        input.value = format(value);
        return;
      }
      const bounded = clampParameter(parsed);
      update(bounded);
      input.value = format(bounded);
      updateAfterChange();
    };
    input.addEventListener("input", () => {
      const parsed = Number.parseFloat(input.value);
      if (!Number.isFinite(parsed) || (parsed >= -PARAMETER_LIMIT && parsed <= PARAMETER_LIMIT)) {
        return;
      }
      commitParameter(input.value);
    });
    input.addEventListener("change", () => {
      commitParameter(input.value);
    });
    return input;
  }

  function makeMatrixSymbol(base, layer) {
    const symbol = createElement("span", "nn-matrix-symbol");
    const letter = createElement("i");
    letter.textContent = base;
    const superscript = document.createElement("sup");
    superscript.textContent = `[${layer}]`;
    symbol.append(letter, superscript);
    return symbol;
  }

  function makeMatrixEditor({ base, layer, rows, columns, entries }) {
    const editor = createElement("div", "nn-matrix-editor nn-matrix-input-editor");
    const symbol = makeMatrixSymbol(base, layer);
    const equals = createElement("span", "nn-matrix-equals");
    equals.textContent = "=";
    const brackets = createElement("div", "nn-matrix-brackets");
    const grid = createElement("div", "nn-matrix-grid");
    grid.style.gridTemplateColumns = `repeat(${columns}, minmax(0, 1fr))`;
    grid.style.gridTemplateRows = `repeat(${rows}, auto)`;
    entries.forEach((entry) => grid.append(entry));
    brackets.append(grid);
    editor.append(symbol, equals, brackets);
    return editor;
  }

  function makeMatrixValueDisplay({ base, layer, rows, columns, values, annotation }) {
    const display = createElement("div", "nn-matrix-editor nn-matrix-value-display");
    const symbol = makeMatrixSymbol(base, layer);
    const equals = createElement("span", "nn-matrix-equals");
    equals.textContent = "=";
    const brackets = createElement("div", "nn-matrix-brackets");
    const grid = createElement("div", "nn-matrix-grid");
    grid.style.gridTemplateColumns = `repeat(${columns}, minmax(0, 1fr))`;
    grid.style.gridTemplateRows = `repeat(${rows}, auto)`;
    values.forEach((value) => {
      const cell = createElement("span", "nn-matrix-value");
      cell.textContent = format(value);
      grid.append(cell);
    });
    brackets.append(grid);
    const note = createElement("span", "nn-matrix-shape");
    note.textContent = annotation;
    display.append(symbol, equals, brackets, note);
    return display;
  }

  function hiddenWeightEntries() {
    return state.weightsHidden.flatMap((row, hiddenIndex) => row.map((weight, inputIndex) => (
      parameterInput({
        label: `Entry ${hiddenIndex + 1}, ${inputIndex + 1} of W superscript [1]`,
        value: weight,
        update: (next) => { state.weightsHidden[hiddenIndex][inputIndex] = next; },
        rowClass: `nn-parameter-row-${hiddenIndex + 1}`,
      })
    )));
  }

  function hiddenBiasEntries() {
    return state.biasHidden.map((bias, hiddenIndex) => parameterInput({
      label: `Entry ${hiddenIndex + 1} of B superscript [1]`,
      value: bias,
      update: (next) => { state.biasHidden[hiddenIndex] = next; },
      rowClass: `nn-parameter-row-${hiddenIndex + 1}`,
    }));
  }

  function outputWeightEntries() {
    return state.weightsOutput.map((weight, hiddenIndex) => parameterInput({
      label: `Entry 1, ${hiddenIndex + 1} of W superscript [2]`,
      value: weight,
      update: (next) => { state.weightsOutput[hiddenIndex] = next; },
      rowClass: "nn-parameter-row-output",
    }));
  }

  function outputBiasEntries() {
    return [parameterInput({
      label: "Entry 1 of B superscript [2]",
      value: state.biasOutput,
      update: (next) => { state.biasOutput = next; },
      rowClass: "nn-parameter-row-output",
    })];
  }

  function renderControls() {
    inputControls.replaceChildren(makeInputControl(0), makeInputControl(1));
    parameterControls.replaceChildren();
    const hiddenCard = createElement("section", "nn-parameter-card");
    const hiddenTitle = createElement("h3");
    hiddenTitle.textContent = "Layer 1 (hidden layer)";
    const hiddenDescription = createElement("p");
    hiddenDescription.textContent = "Each colored row controls the like-colored input connections; A is the value after ReLU.";
    const hiddenMatrices = createElement("div", "nn-matrix-pair nn-layer-one-matrices");
    const hiddenActivation = createElement("div", "nn-hidden-activation");
    hiddenActivation.id = "nn-hidden-activation";
    hiddenMatrices.append(
      makeMatrixEditor({
        base: "W",
        layer: 1,
        rows: state.weightsHidden.length,
        columns: state.input.length,
        entries: hiddenWeightEntries(),
      }),
      makeMatrixEditor({
        base: "B",
        layer: 1,
        rows: state.biasHidden.length,
        columns: 1,
        entries: hiddenBiasEntries(),
      }),
      hiddenActivation,
    );
    hiddenCard.append(
      hiddenTitle,
      hiddenDescription,
      hiddenMatrices,
    );
    const outputCard = createElement("section", "nn-parameter-card");
    const outputTitle = createElement("h3");
    outputTitle.textContent = "Layer 2 (output layer)";
    const outputDescription = createElement("p");
    outputDescription.textContent = "One sigmoid output unit";
    const outputMatrices = createElement("div", "nn-matrix-pair");
    outputMatrices.append(
      makeMatrixEditor({
        base: "W",
        layer: 2,
        rows: 1,
        columns: state.weightsOutput.length,
        entries: outputWeightEntries(),
      }),
      makeMatrixEditor({
        base: "B",
        layer: 2,
        rows: 1,
        columns: 1,
        entries: outputBiasEntries(),
      }),
    );
    outputCard.append(
      outputTitle,
      outputDescription,
      outputMatrices,
    );
    parameterControls.append(hiddenCard, outputCard);
  }

  function renderDatasetControls() {
    datasetControls.replaceChildren();
    Object.entries(DATASETS).forEach(([key, dataset]) => {
      const button = createElement(
        "button",
        `nn-dataset-button${selectedDataset === key ? " nn-dataset-button-active" : ""}`,
      );
      button.type = "button";
      button.textContent = dataset.label;
      button.setAttribute("aria-pressed", String(selectedDataset === key));
      button.addEventListener("click", () => {
        selectedDataset = key;
        renderDatasetControls();
        renderProbabilityPlane();
        status.textContent = `${dataset.label} selected`;
      });
      datasetControls.append(button);
    });
  }

  function renderHiddenActivation() {
    const activationDisplay = root.querySelector("#nn-hidden-activation");
    if (!activationDisplay) return;
    const output = forward(state.input);
    activationDisplay.replaceChildren(
      makeMatrixValueDisplay({
        base: "A",
        layer: 1,
        rows: state.biasHidden.length,
        columns: 1,
        values: output.hiddenActivations,
        annotation: "after ReLU",
      }),
    );
  }

  function planeCoordinate(value, start, size) {
    return start + (value - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0]) * size;
  }

  function renderProbabilityPlane() {
    probabilityPlane.replaceChildren();
    probabilityPlane.setAttribute("viewBox", "0 0 420 380");
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
    probabilityPlane.append(grid);
    const xAxis = svgElement("line", {
      x1: left, y1: top + height, x2: left + width, y2: top + height,
      class: "nn-axis-line",
    });
    const yAxis = svgElement("line", {
      x1: left, y1: top, x2: left, y2: top + height,
      class: "nn-axis-line",
    });
    probabilityPlane.append(xAxis, yAxis);
    [-2, -1, 0, 1, 2].forEach((tick) => {
      const x = planeCoordinate(tick, left, width);
      const y = top + height - (tick - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0]) * height;
      probabilityPlane.append(
        svgElement("line", { x1: x, y1: top, x2: x, y2: top + height, class: "nn-grid-line" }),
        svgElement("line", { x1: left, y1: y, x2: left + width, y2: y, class: "nn-grid-line" }),
      );
      addSvgText(probabilityPlane, String(tick), x, top + height + 20, {
        class: "nn-axis-text", "text-anchor": "middle",
      });
      addSvgText(probabilityPlane, String(tick), left - 10, y + 4, {
        class: "nn-axis-text", "text-anchor": "end",
      });
    });
    addSvgText(probabilityPlane, "x₁", left + width / 2, 372, {
      class: "nn-axis-label", "text-anchor": "middle",
    });
    addSvgText(probabilityPlane, "x₂", 15, top + height / 2, {
      class: "nn-axis-label", transform: `rotate(-90 15 ${top + height / 2})`, "text-anchor": "middle",
    });
    const pointsLayer = svgElement("g", { class: "nn-data-points" });
    DATASETS[selectedDataset].points.forEach((point) => {
      const circle = svgElement("circle", {
        cx: planeCoordinate(point.x, left, width),
        cy: top + height - (point.y - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0]) * height,
        r: 4.5,
        class: `nn-data-point nn-data-point-${point.label === 1 ? "positive" : "negative"}`,
      });
      const title = svgElement("title");
      title.textContent = `Toy observation: label ${point.label}`;
      circle.append(title);
      pointsLayer.append(circle);
    });
    probabilityPlane.append(pointsLayer);
    const markerX = planeCoordinate(state.input[0], left, width);
    const markerY = top + height - (state.input[1] - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0]) * height;
    probabilityPlane.append(
      svgElement("circle", { cx: markerX, cy: markerY, r: 7, class: "nn-input-marker-outer" }),
      svgElement("circle", { cx: markerX, cy: markerY, r: 3.6, class: "nn-input-marker-inner" }),
    );
    addSvgText(probabilityPlane, "selected input", markerX + 10, markerY - 10, {
      class: "nn-marker-label",
    });
  }

  function drawEdge(svg, from, to, weight, colorClass, active) {
    const line = svgElement("line", {
      x1: from.x, y1: from.y, x2: to.x, y2: to.y,
      "stroke-width": weightWidth(weight),
      class: `nn-network-edge ${colorClass}${active ? " nn-network-edge-active" : ""}`,
    });
    svg.append(line);
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

  function drawNode(svg, position, lines, nodeClass, active, width = 122) {
    const group = svgElement("g", { class: `nn-network-node ${nodeClass}${active ? " nn-network-node-active" : ""}` });
    const box = svgElement("rect", {
      x: position.x - width / 2, y: position.y - 31, width, height: 62, rx: 9,
    });
    group.append(box);
    lines.forEach((line, index) => {
      const text = svgElement("text", {
        x: position.x,
        y: position.y - 8 + index * 17,
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

  function renderNetworkDiagram() {
    const output = forward(state.input);
    networkDiagram.replaceChildren();
    networkDiagram.setAttribute("viewBox", "0 0 750 450");
    const positions = {
      inputs: [{ x: 100, y: 120 }, { x: 100, y: 330 }],
      hidden: [{ x: 380, y: 80 }, { x: 380, y: 225 }, { x: 380, y: 370 }],
      output: { x: 650, y: 225 },
    };
    addSvgText(networkDiagram, "input", 100, 28, { class: "nn-layer-title", "text-anchor": "middle" });
    addSvgText(networkDiagram, "hidden layer (ReLU)", 380, 28, { class: "nn-layer-title", "text-anchor": "middle" });
    addSvgText(networkDiagram, "output (sigmoid)", 650, 28, { class: "nn-layer-title", "text-anchor": "middle" });
    output.hiddenContributions.forEach((contributions, hiddenIndex) => {
      contributions.forEach((_contribution, inputIndex) => {
        drawEdge(
          networkDiagram,
          positions.inputs[inputIndex],
          positions.hidden[hiddenIndex],
          state.weightsHidden[hiddenIndex][inputIndex],
          `nn-network-edge-hidden-${hiddenIndex + 1}`,
          animationPhase === 1,
        );
      });
    });
    output.outputContributions.forEach((_contribution, hiddenIndex) => {
      drawEdge(
        networkDiagram,
        positions.hidden[hiddenIndex],
        positions.output,
        state.weightsOutput[hiddenIndex],
        "nn-network-edge-output",
        animationPhase === 2,
      );
    });
    positions.inputs.forEach((position, index) => {
      drawNode(
        networkDiagram,
        position,
        [{ base: "x", subscript: String(index + 1) }, format(state.input[index])],
        "nn-input-node",
        animationPhase === 1,
      );
    });
    positions.hidden.forEach((position, index) => {
      drawNode(
        networkDiagram,
        position,
        [
          {
            base: "z",
            subscript: String(index + 1),
            superscript: "[1]",
            suffix: ` = ${format(output.hiddenScores[index])}`,
          },
          {
            base: "a",
            subscript: String(index + 1),
            superscript: "[1]",
            suffix: ` = ${format(output.hiddenActivations[index])}`,
          },
        ],
        "nn-hidden-node",
        animationPhase === 2,
      );
    });
    drawNode(
      networkDiagram,
      positions.output,
      [`z = ${format(output.outputScore)}`, `p = ${format(output.probability, 3)}`],
      output.probability >= 0.5 ? "nn-output-positive" : "nn-output-negative",
      animationPhase === 3,
    );
  }

  function renderViews() {
    renderHiddenActivation();
    renderProbabilityPlane();
    renderNetworkDiagram();
  }

  function stopAnimation() {
    if (animationTimer !== null) {
      window.clearTimeout(animationTimer);
      animationTimer = null;
    }
    animationPhase = 0;
    animateButton.disabled = false;
    status.textContent = "Ready";
  }

  function animateForwardPass() {
    stopAnimation();
    animateButton.disabled = true;
    const phases = [
      { phase: 1, text: "1 of 3: inputs contribute to the hidden units" },
      { phase: 2, text: "2 of 3: hidden activations contribute to the output" },
      { phase: 3, text: "3 of 3: sigmoid produces the probability" },
    ];
    let index = 0;
    const advance = () => {
      const current = phases[index];
      animationPhase = current.phase;
      status.textContent = current.text;
      renderNetworkDiagram();
      index += 1;
      if (index < phases.length) {
        animationTimer = window.setTimeout(advance, 900);
      } else {
        animationTimer = window.setTimeout(() => {
          animationPhase = 0;
          animateButton.disabled = false;
          status.textContent = "Forward pass complete";
          renderNetworkDiagram();
          animationTimer = null;
        }, 1100);
      }
    };
    advance();
  }

  probabilityPlane.addEventListener("click", (event) => {
    const bounds = probabilityPlane.getBoundingClientRect();
    const x = DOMAIN[0] + (event.clientX - bounds.left) / bounds.width
      * (DOMAIN[1] - DOMAIN[0]);
    const y = DOMAIN[1] - (event.clientY - bounds.top) / bounds.height
      * (DOMAIN[1] - DOMAIN[0]);
    state.input = [clampInput(x), clampInput(y)];
    renderControls();
    updateAfterChange();
    status.textContent = "Input selected from the data plane";
  });

  animateButton.addEventListener("click", animateForwardPass);
  resetButton.addEventListener("click", () => {
    state = cloneState(DEFAULT_STATE);
    selectedDataset = "moons";
    stopAnimation();
    renderDatasetControls();
    renderControls();
    renderViews();
    status.textContent = "Example restored";
  });

  renderDatasetControls();
  renderControls();
  renderViews();
})();
