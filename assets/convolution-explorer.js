(() => {
  const root = document.getElementById("convolution-explorer");
  if (!root) return;

  const mainChart = root.querySelector("#convolution-main-chart");
  const calculation = root.querySelector("#convolution-calculation");
  const kernelDescription = root.querySelector("#convolution-kernel-description");
  const responseInsight = root.querySelector("#convolution-response-insight");
  const status = root.querySelector("#convolution-status");
  const previousButton = root.querySelector("#convolution-previous");
  const nextButton = root.querySelector("#convolution-next");
  const playButton = root.querySelector("#convolution-play");
  const resetButton = root.querySelector("#convolution-reset");
  const kernelButtons = [...root.querySelectorAll("[data-kernel]")];
  const kernelThumbnails = [...root.querySelectorAll("[data-kernel-thumbnail]")];

  if (!mainChart || !calculation || !kernelDescription || !responseInsight || !status
    || !previousButton || !nextButton || !playButton || !resetButton) return;

  const SVG_NS = "http://www.w3.org/2000/svg";
  const SIGNAL = Array.from({ length: 121 }, (_, time) => {
    const baseline = 0.23 + 0.016 * Math.sin(0.29 * time) + 0.009 * Math.sin(1.07 * time);
    const rise = 0.73 / (1 + Math.exp(-(time - 35) / 1.65));
    const fall = 0.58 / (1 + Math.exp(-(time - 79) / 1.85));
    const localDip = -0.055 * Math.exp(-(((time - 16) / 2.4) ** 2));
    const localPeak = 0.10 * Math.exp(-(((time - 104) / 1.8) ** 2));
    const transientBurst = 0.085 * Math.exp(-(((time - 96) / 4.2) ** 2)) * Math.sin(3.5 * time);
    return baseline + rise - fall + localDip + localPeak + transientBurst;
  });
  const RADIUS = 2;
  const POSITIONS = Array.from(
    { length: SIGNAL.length - 2 * RADIUS },
    (_, index) => index + RADIUS,
  );
  const KERNELS = {
    average: {
      label: "Smooth local noise",
      weights: [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
      description: "A five-point running average gives equal weight to five neighboring sensor values, so local wiggles are softened.",
      insight: "The lower trace is the five-sample running average: it follows the slower trend while smoothing the faster wiggles in the input.",
    },
    change: {
      label: "Detect rising / falling changes",
      weights: [-1 / 6, -2 / 6, 0, 2 / 6, 1 / 6],
      description: "A five-point change detector compares values on both sides of the center while smoothing small wiggles.",
      insight: "Large positive values in the lower trace mark a local rise; large negative values mark a local fall. Flat regions stay near zero.",
    },
    peak: {
      label: "Detect local peaks / dips",
      weights: [-1 / 4, -1 / 4, 1, -1 / 4, -1 / 4],
      description: "A five-point curvature filter compares the center sample with four nearby values.",
      insight: "Positive values in the lower trace identify local peaks; negative values identify local dips relative to nearby samples.",
    },
  };
  const WEIGHT_COLORS = ["#c7443e", "#d37a70", "#6f7c84", "#72a880", "#21804d"];
  const state = { kernel: "change", positionIndex: 0, timer: null };

  function svgElement(name, attributes = {}) {
    const element = document.createElementNS(SVG_NS, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
    return element;
  }

  function appendSvgText(svg, text, x, y, attributes = {}) {
    const element = svgElement("text", { x, y, ...attributes });
    element.textContent = text;
    svg.append(element);
    return element;
  }

  function makePath(points) {
    return points.map(([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  }

  function makeSmoothPath(points) {
    if (points.length < 2) return "";
    let path = `M${points[0][0].toFixed(2)},${points[0][1].toFixed(2)}`;
    for (let index = 1; index < points.length - 1; index += 1) {
      const [x, y] = points[index];
      const [nextX, nextY] = points[index + 1];
      path += ` Q${x.toFixed(2)},${y.toFixed(2)} ${((x + nextX) / 2).toFixed(2)},${((y + nextY) / 2).toFixed(2)}`;
    }
    const [lastX, lastY] = points[points.length - 1];
    path += ` L${lastX.toFixed(2)},${lastY.toFixed(2)}`;
    return path;
  }

  function format(value, digits = 2) {
    const rounded = Number(value.toFixed(digits));
    return Math.abs(rounded) < 10 ** (-digits) ? "0" : rounded.toFixed(digits);
  }

  function signedFormat(value, digits = 2) {
    const rendered = format(value, digits);
    return Number(value) > 0 ? `+${rendered}` : rendered;
  }

  function currentPosition() {
    return POSITIONS[state.positionIndex];
  }

  function currentKernel() {
    return KERNELS[state.kernel];
  }

  function responseAt(position) {
    return currentKernel().weights.reduce(
      (sum, weight, offset) => sum + weight * SIGNAL[position + offset - RADIUS],
      0,
    );
  }

  function stopPlayback() {
    if (state.timer !== null) {
      window.clearInterval(state.timer);
      state.timer = null;
    }
  }

  function isPlaying() {
    return state.timer !== null;
  }

  function renderKernelThumbnails() {
    kernelThumbnails.forEach((thumbnail) => {
      const kernel = KERNELS[thumbnail.dataset.kernelThumbnail];
      const width = 96;
      const height = 38;
      const maxMagnitude = Math.max(...kernel.weights.map((weight) => Math.abs(weight)));
      const x = (index) => 9 + (index / (kernel.weights.length - 1)) * (width - 18);
      const y = (weight) => height / 2 - (weight / maxMagnitude) * 12;
      thumbnail.setAttribute("viewBox", `0 0 ${width} ${height}`);
      thumbnail.replaceChildren();
      thumbnail.append(svgElement("path", {
        d: makeSmoothPath(kernel.weights.map((weight, index) => [x(index), y(weight)])),
        fill: "none",
        stroke: "currentColor",
        "stroke-width": 2.5,
        "stroke-linejoin": "round",
        "stroke-linecap": "round",
      }));
    });
  }

  function renderButtons() {
    const atStart = state.positionIndex === 0;
    const atEnd = state.positionIndex === POSITIONS.length - 1;
    previousButton.disabled = atStart || isPlaying();
    nextButton.disabled = atEnd || isPlaying();
    resetButton.disabled = isPlaying();
    playButton.disabled = atEnd && !isPlaying();
    playButton.textContent = isPlaying() ? "Pause" : "Play all";
    kernelButtons.forEach((button) => {
      const active = button.dataset.kernel === state.kernel;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
      button.disabled = isPlaying();
    });
  }

  function renderStatus() {
    const completed = state.positionIndex + 1;
    status.textContent = `Window ${completed} of ${POSITIONS.length}: centered at t = ${currentPosition()}`;
  }

  function addAxes(svg, { x1, x2, y1, y2, x, y, ticks, label, showXLabels = false }) {
    ticks.forEach((tick) => {
      svg.append(svgElement("line", {
        x1, x2, y1: y(tick), y2: y(tick),
        stroke: tick === 0 ? "#a7b2b8" : "#d9e2e6",
        "stroke-width": tick === 0 ? 1.2 : 1,
        "stroke-dasharray": tick === 0 ? "0" : "4 5",
      }));
      appendSvgText(svg, format(tick), x1 - 10, y(tick) + 4, {
        "text-anchor": "end", "font-size": 11, fill: "#65737b",
      });
    });
    svg.append(svgElement("line", { x1, x2, y1: y2, y2, stroke: "#728088", "stroke-width": 1.2 }));
    svg.append(svgElement("line", { x1, x2: x1, y1, y2, stroke: "#728088", "stroke-width": 1.2 }));
    appendSvgText(svg, label, 18, (y1 + y2) / 2, {
      transform: `rotate(-90 18 ${(y1 + y2) / 2})`, "text-anchor": "middle",
      "font-size": 12, fill: "#42535b", "font-weight": 650,
    });
    if (showXLabels) {
      for (let tick = 0; tick <= SIGNAL.length - 1; tick += 30) {
        appendSvgText(svg, String(tick), x(tick), y2 + 20, {
          "text-anchor": "middle", "font-size": 11, fill: "#65737b",
        });
      }
      appendSvgText(svg, "sample time, t", (x1 + x2) / 2, y2 + 38, {
        "text-anchor": "middle", "font-size": 12, fill: "#42535b", "font-weight": 650,
      });
    }
  }

  function renderMainChart() {
    const width = 900;
    const height = 535;
    const margin = { right: 28, left: 70 };
    const plotWidth = width - margin.left - margin.right;
    const topPlot = { y1: 58, y2: 294, yMin: -0.45, yMax: 1.15 };
    const bottomPlot = { y1: 366, y2: 488 };
    const position = currentPosition();
    const kernel = currentKernel();
    const responses = POSITIONS.map((time) => responseAt(time));
    const responseLimit = Math.max(
      0.04,
      Math.ceil(Math.max(...responses.map((value) => Math.abs(value))) * 1.2 * 100) / 100,
    );
    const x = (index) => margin.left + (index / (SIGNAL.length - 1)) * plotWidth;
    const yTop = (value) => topPlot.y1 + ((topPlot.yMax - value) / (topPlot.yMax - topPlot.yMin)) * (topPlot.y2 - topPlot.y1);
    const yBottom = (value) => bottomPlot.y1 + ((responseLimit - value) / (2 * responseLimit)) * (bottomPlot.y2 - bottomPlot.y1);
    const completed = responses.slice(0, state.positionIndex + 1);

    mainChart.setAttribute("viewBox", `0 0 ${width} ${height}`);
    mainChart.replaceChildren();
    const title = svgElement("title");
    title.textContent = `Sensor trace with an overlaid moving five-point kernel and convolution response for a window centered at t equals ${position}`;
    mainChart.append(title);
    const description = svgElement("desc");
    description.textContent = "The upper plot overlays the original sensor trace with the five kernel weights. The kernel moves horizontally with the colored signal values. The lower plot is the convolution response calculated so far.";
    mainChart.append(description);

    appendSvgText(mainChart, "Sensor trace with moving kernel", margin.left, 29, {
      "font-size": 15, fill: "#2f7395", "font-weight": 750,
    });
    appendSvgText(mainChart, "Convolution response", margin.left, 338, {
      "font-size": 15, fill: "#a84e17", "font-weight": 750,
    });

    const windowLeft = x(position - RADIUS) - 6;
    const windowRight = x(position + RADIUS) + 6;
    mainChart.append(svgElement("rect", {
      x: windowLeft, y: topPlot.y1 - 4, width: windowRight - windowLeft,
      height: topPlot.y2 - topPlot.y1 + 8, fill: "#e5eff5", rx: 4,
    }));

    addAxes(mainChart, {
      x1: margin.left, x2: width - margin.right, y1: topPlot.y1, y2: topPlot.y2,
      x, y: yTop, ticks: [-0.25, 0, 0.5, 1], label: "value",
    });
    addAxes(mainChart, {
      x1: margin.left, x2: width - margin.right, y1: bottomPlot.y1, y2: bottomPlot.y2,
      x, y: yBottom, ticks: [-responseLimit, 0, responseLimit],
      label: "response, rₜ", showXLabels: true,
    });

    mainChart.append(svgElement("path", {
      d: makePath(SIGNAL.map((value, index) => [x(index), yTop(value)])),
      fill: "none", stroke: "#3478a6", "stroke-width": 2.2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    }));
    SIGNAL.forEach((value, index) => {
      const offset = index - (position - RADIUS);
      const active = offset >= 0 && offset < kernel.weights.length;
      mainChart.append(svgElement("circle", {
        cx: x(index), cy: yTop(value), r: active ? 5.7 : 1.8,
        fill: active ? WEIGHT_COLORS[offset] : "#3478a6",
        stroke: active ? "#fff" : "none", "stroke-width": active ? 1.4 : 0,
      }));
    });
    const kernelPoints = kernel.weights.map((weight, index) => [
      x(position - RADIUS + index), yTop(weight),
    ]);
    mainChart.append(svgElement("path", {
      d: makeSmoothPath(kernelPoints),
      fill: "none", stroke: "#785c8d", "stroke-width": 3.2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    }));

    if (completed.length > 1) {
      mainChart.append(svgElement("path", {
        d: makePath(completed.map((value, index) => [x(POSITIONS[index]), yBottom(value)])),
        fill: "none", stroke: "#dd6b20", "stroke-width": 2.6,
        "stroke-linejoin": "round", "stroke-linecap": "round",
      }));
    }
    completed.forEach((value, index) => {
      const time = POSITIONS[index];
      const active = time === position;
      mainChart.append(svgElement("circle", {
        cx: x(time), cy: yBottom(value), r: active ? 5.9 : 2.9,
        fill: active ? "#dd6b20" : "#f2a366",
        stroke: active ? "#fff" : "none", "stroke-width": active ? 1.5 : 0,
      }));
    });
    const response = responseAt(position);
    appendSvgText(mainChart, `r${position} = ${signedFormat(response)}`, x(position), Math.max(bottomPlot.y1 + 13, yBottom(response) - 11), {
      "text-anchor": "middle", "font-size": 12, fill: "#a84e17", "font-weight": 750,
    });
  }

  function renderCalculation() {
    const position = currentPosition();
    const kernel = currentKernel();
    const offsets = Array.from({ length: kernel.weights.length }, (_, index) => index - RADIUS);
    const values = offsets.map((offset) => SIGNAL[position + offset]);
    const contributions = kernel.weights.map((weight, index) => weight * values[index]);
    const response = contributions.reduce((sum, contribution) => sum + contribution, 0);
    const headers = offsets.map((offset) => {
      if (offset === 0) return "t";
      return offset < 0 ? `t − ${Math.abs(offset)}` : `t + ${offset}`;
    });

    calculation.replaceChildren();
    const table = document.createElement("table");
    table.className = "convolution-table";
    table.innerHTML = `
      <caption>Window centered at sample time t = ${position}</caption>
      <thead><tr><th scope="col"></th>${headers.map((header) => `<th scope="col">${header}</th>`).join("")}</tr></thead>
      <tbody>
        <tr><th scope="row">sensor value</th>${values.map((value) => `<td>${format(value)}</td>`).join("")}</tr>
        <tr><th scope="row">kernel weight</th>${kernel.weights.map((weight, index) => `<td><span class="convolution-weight-value" style="--weight-color: ${WEIGHT_COLORS[index]}">${signedFormat(weight)}</span></td>`).join("")}</tr>
        <tr><th scope="row">final value</th>${contributions.map((value, index) => `<td><span style="color: ${WEIGHT_COLORS[index]}; font-weight: 700;">${signedFormat(value)}</span></td>`).join("")}</tr>
      </tbody>
    `;
    calculation.append(table);

    const formula = document.createElement("div");
    formula.className = "convolution-formula";
    formula.innerHTML = `<span><i>r</i><sub>${position}</sub> = ${contributions.map((value) => `(${signedFormat(value)})`).join(" + ")}</span><strong>= ${signedFormat(response)}</strong>`;
    calculation.append(formula);
  }

  function renderAll() {
    const kernel = currentKernel();
    kernelDescription.textContent = kernel.description;
    responseInsight.textContent = kernel.insight;
    renderKernelThumbnails();
    renderButtons();
    renderStatus();
    renderMainChart();
    renderCalculation();
  }

  function moveBy(amount) {
    stopPlayback();
    state.positionIndex = Math.max(0, Math.min(POSITIONS.length - 1, state.positionIndex + amount));
    renderAll();
  }

  function reset() {
    stopPlayback();
    state.positionIndex = 0;
    renderAll();
  }

  kernelButtons.forEach((button) => {
    button.addEventListener("click", () => {
      stopPlayback();
      state.kernel = button.dataset.kernel;
      state.positionIndex = 0;
      renderAll();
    });
  });
  previousButton.addEventListener("click", () => moveBy(-1));
  nextButton.addEventListener("click", () => moveBy(1));
  resetButton.addEventListener("click", reset);
  playButton.addEventListener("click", () => {
    if (isPlaying()) {
      stopPlayback();
      renderAll();
      return;
    }
    if (state.positionIndex === POSITIONS.length - 1) state.positionIndex = 0;
    state.timer = window.setInterval(() => {
      if (state.positionIndex >= POSITIONS.length - 1) {
        stopPlayback();
        renderAll();
        return;
      }
      state.positionIndex += 1;
      renderAll();
    }, 110);
    renderAll();
  });

  window.addEventListener("pagehide", stopPlayback, { once: true });
  renderAll();
})();
