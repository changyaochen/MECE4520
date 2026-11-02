(() => {
  const root = document.getElementById("ar-order-explorer");
  const data = window.AR_ORDER_EXPLORER_DATA;
  if (!root || !data) return;

  const chart = root.querySelector("#ar-order-chart");
  const chartCopy = root.querySelector("#ar-order-chart-copy");
  const detail = root.querySelector("#ar-order-detail");
  const zoomIn = root.querySelector("#ar-order-zoom-in");
  const zoomOut = root.querySelector("#ar-order-zoom-out");
  const zoomReset = root.querySelector("#ar-order-zoom-reset");
  const focusTest = root.querySelector("#ar-order-focus-test");
  const orderButtons = [...root.querySelectorAll("[data-ar-order]")];
  const scenarioButtons = [...root.querySelectorAll("[data-ar-scenario]")];
  if (!chart || !chartCopy || !detail || !zoomIn || !zoomOut || !zoomReset || !focusTest) return;

  const SVG_NS = "http://www.w3.org/2000/svg";
  const orders = Object.keys(data.models).map(Number).sort((left, right) => left - right);
  const state = {
    order: 1,
    scenario: "easy",
    start: 0,
    end: data.values.length - 1,
  };

  function palette() {
    const dark = document.documentElement.classList.contains("quarto-dark")
      || document.body.classList.contains("quarto-dark");
    return dark
      ? {
        panel: "#20272b", frame: "#56616a", grid: "#465159", text: "#e5ecef", muted: "#c8d1d6",
        observed: "#88c6e8", prediction: "#f0ad5d", train: "#294b5d", test: "#553a24",
      }
      : {
        panel: "#ffffff", frame: "#d6e0e4", grid: "#d9e2e6", text: "#34454d", muted: "#5d6a72",
        observed: "#2f7395", prediction: "#dd6b20", train: "#e5eff5", test: "#fff0e2",
      };
  }

  function svgElement(name, attributes = {}) {
    const element = document.createElementNS(SVG_NS, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
    return element;
  }

  function text(svg, value, x, y, attributes = {}) {
    const element = svgElement("text", { x, y, ...attributes });
    element.textContent = value;
    svg.append(element);
  }

  function path(points) {
    return points.map(([x, y], index) => (index === 0 ? "M" : "L") + x.toFixed(2) + "," + y.toFixed(2)).join(" ");
  }

  function format(value, digits = 2) {
    return Number(value).toFixed(digits);
  }

  function lookbackLabel(order) {
    if (order === 1) return "1 hour";
    if (order === 24) return "1 day";
    if (order === 168) return "1 week";
    return order + " hours";
  }

  function scenarioLabel() {
    return state.scenario === "easy" ? "Easy: actual test lags" : "Difficult: recursive rollout";
  }

  function selectedModel() {
    return data.models[String(state.order)];
  }

  function selectedPredictions() {
    return state.scenario === "easy"
      ? selectedModel().oneStepPredictions
      : selectedModel().recursivePredictions;
  }

  function selectedRmse() {
    return state.scenario === "easy"
      ? selectedModel().oneStepRmse
      : selectedModel().recursiveRmse;
  }

  function setViewBox(svg, width, height, title, description) {
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    svg.replaceChildren();
    const svgTitle = svgElement("title");
    svgTitle.textContent = title;
    svg.append(svgTitle);
    const svgDescription = svgElement("desc");
    svgDescription.textContent = description;
    svg.append(svgDescription);
  }

  function addYAxis(svg, { x, right, top, bottom, values, scale, colors, label }) {
    values.forEach((value) => {
      const y = scale(value);
      svg.append(svgElement("line", {
        x1: x, x2: right, y1: y, y2: y, stroke: colors.grid, "stroke-width": 1,
      }));
      text(svg, format(value, 1), x - 9, y + 4, {
        "text-anchor": "end", "font-size": 10.5, fill: colors.muted,
      });
    });
    svg.append(svgElement("line", { x1: x, x2: x, y1: top, y2: bottom, stroke: colors.muted, "stroke-width": 1.2 }));
    svg.append(svgElement("line", { x1: x, x2: right, y1: bottom, y2: bottom, stroke: colors.muted, "stroke-width": 1.2 }));
    text(svg, label, 18, (top + bottom) / 2, {
      transform: "rotate(-90 18 " + (top + bottom) / 2 + ")", "text-anchor": "middle",
      "font-size": 12, fill: colors.text, "font-weight": 700,
    });
  }

  function visiblePredictionPoints(start, end) {
    const predictions = selectedPredictions();
    const first = Math.max(start, data.trainEnd);
    return Array.from({ length: Math.max(0, end - first + 1) }, (_, offset) => {
      const index = first + offset;
      return [index, predictions[index]];
    }).filter(([, value]) => value !== null);
  }

  function renderChart() {
    const colors = palette();
    const width = 960;
    const height = 440;
    const margin = { left: 65, right: 27, top: 42, bottom: 58 };
    const right = width - margin.right;
    const bottom = height - margin.bottom;
    const start = state.start;
    const end = state.end;
    const observed = data.values.slice(start, end + 1);
    const predictionPoints = visiblePredictionPoints(start, end);
    const chartValues = [...observed, ...predictionPoints.map(([, value]) => value)];
    const lower = Math.min(...chartValues);
    const upper = Math.max(...chartValues);
    const padding = Math.max(0.7, (upper - lower) * 0.14);
    const yMin = lower - padding;
    const yMax = upper + padding;
    const x = (index) => margin.left + ((index - start) / Math.max(1, end - start)) * (right - margin.left);
    const y = (value) => margin.top + ((yMax - value) / (yMax - yMin)) * (bottom - margin.top);
    const splitVisible = data.trainEnd >= start && data.trainEnd <= end;
    const splitX = x(data.trainEnd - 0.5);

    setViewBox(
      chart,
      width,
      height,
      "AR " + state.order + " " + scenarioLabel() + " forecast",
      "Observed oil temperature with a chronological train-test split. The orange line is the selected autoregressive forecast over the test period.",
    );
    chart.append(svgElement("rect", {
      x: margin.left, y: margin.top, width: right - margin.left, height: bottom - margin.top,
      fill: colors.panel, stroke: colors.frame, "stroke-width": 1,
    }));
    if (splitVisible) {
      chart.append(svgElement("rect", {
        x: margin.left, y: margin.top, width: splitX - margin.left, height: bottom - margin.top, fill: colors.train,
      }));
      chart.append(svgElement("rect", {
        x: splitX, y: margin.top, width: right - splitX, height: bottom - margin.top, fill: colors.test,
      }));
    } else if (start >= data.trainEnd) {
      chart.append(svgElement("rect", {
        x: margin.left, y: margin.top, width: right - margin.left, height: bottom - margin.top, fill: colors.test,
      }));
    } else {
      chart.append(svgElement("rect", {
        x: margin.left, y: margin.top, width: right - margin.left, height: bottom - margin.top, fill: colors.train,
      }));
    }
    addYAxis(chart, {
      x: margin.left, right, top: margin.top, bottom,
      values: [yMin, (yMin + yMax) / 2, yMax], scale: y, colors, label: "OT (°C)",
    });
    chart.append(svgElement("path", {
      d: path(observed.map((value, offset) => [x(start + offset), y(value)])), fill: "none", stroke: colors.observed,
      "stroke-width": 2.1, "stroke-linejoin": "round", "stroke-linecap": "round",
    }));
    if (predictionPoints.length > 1) {
      chart.append(svgElement("path", {
        d: path(predictionPoints.map(([index, value]) => [x(index), y(value)])), fill: "none", stroke: colors.prediction,
        "stroke-width": 2.1, "stroke-dasharray": "6 3", "stroke-linejoin": "round", "stroke-linecap": "round",
      }));
    }
    if (splitVisible) {
      chart.append(svgElement("line", {
        x1: splitX, x2: splitX, y1: margin.top, y2: bottom,
        stroke: colors.prediction, "stroke-width": 1.8, "stroke-dasharray": "6 4",
      }));
      text(chart, "training", (margin.left + splitX) / 2, margin.top + 20, {
        "text-anchor": "middle", "font-size": 12, fill: colors.text, "font-weight": 700,
      });
      text(chart, "test", (splitX + right) / 2, margin.top + 20, {
        "text-anchor": "middle", "font-size": 12, fill: colors.text, "font-weight": 700,
      });
      text(chart, "split", splitX + 7, margin.top + 40, {
        "font-size": 10.5, fill: colors.prediction, "font-weight": 700,
      });
    }
    const tickIndices = [...new Set([start, Math.round((start + end) / 2), end])];
    tickIndices.forEach((index) => {
      text(chart, data.dates[index], x(index), bottom + 20, {
        "text-anchor": "middle", "font-size": 10.5, fill: colors.muted,
      });
    });
    text(chart, "hourly timestamp", (margin.left + right) / 2, height - 9, {
      "text-anchor": "middle", "font-size": 12, fill: colors.text, "font-weight": 700,
    });
    chart.append(svgElement("line", {
      x1: right - 230, x2: right - 207, y1: margin.top + 17, y2: margin.top + 17,
      stroke: colors.observed, "stroke-width": 2.4,
    }));
    text(chart, "observed OT", right - 200, margin.top + 21, { "font-size": 11, fill: colors.muted });
    chart.append(svgElement("line", {
      x1: right - 112, x2: right - 89, y1: margin.top + 17, y2: margin.top + 17,
      stroke: colors.prediction, "stroke-width": 2.4, "stroke-dasharray": "6 3",
    }));
    text(chart, "AR forecast", right - 82, margin.top + 21, { "font-size": 11, fill: colors.muted });

    const scenarioCopy = state.scenario === "easy"
      ? "Every test-hour forecast receives the actual preceding test values as its lags."
      : "After the split, each new forecast becomes a lagged input for later forecasts.";
    chartCopy.textContent = "Selected model: AR(" + state.order + ") with " + lookbackLabel(state.order)
      + " of history. " + scenarioCopy;
  }

  function renderDetail() {
    const scenarioExplanation = state.scenario === "easy"
      ? "At test hour k, the model uses the actual Xₖ₋₁, Xₖ₋₂, …, Xₖ₋ₚ."
      : "At test hour k, the model uses its own earlier forecasts once it has crossed the split.";
    detail.innerHTML = "<div class=\"ar-order-stat-grid\">"
      + "<div class=\"ar-order-stat\"><span>Selected model</span><strong>AR(" + state.order + ")</strong></div>"
      + "<div class=\"ar-order-stat\"><span>History used</span><strong>" + lookbackLabel(state.order) + "</strong></div>"
      + "<div class=\"ar-order-stat\"><span>Test RMSE</span><strong>" + format(selectedRmse()) + " °C</strong></div>"
      + "</div><div class=\"ar-order-formula\"><span>" + scenarioLabel() + "</span><strong>"
      + scenarioExplanation + "</strong></div>";
  }

  function renderControls() {
    orderButtons.forEach((button) => {
      const active = Number(button.dataset.arOrder) === state.order;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    scenarioButtons.forEach((button) => {
      const active = button.dataset.arScenario === state.scenario;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    const fullRange = data.values.length - 1;
    zoomOut.disabled = state.start === 0 && state.end === fullRange;
    zoomReset.disabled = state.start === 0 && state.end === fullRange;
  }

  function render() {
    renderControls();
    renderChart();
    renderDetail();
  }

  function setRange(start, end) {
    state.start = Math.max(0, Math.round(start));
    state.end = Math.min(data.values.length - 1, Math.round(end));
    render();
  }

  function zoomInChart() {
    const span = state.end - state.start;
    const nextSpan = Math.max(36, Math.round(span * 0.55));
    const center = (state.start + state.end) / 2;
    setRange(center - nextSpan / 2, center + nextSpan / 2);
  }

  function zoomOutChart() {
    const span = state.end - state.start;
    const nextSpan = Math.min(data.values.length - 1, Math.round(span / 0.55));
    const center = (state.start + state.end) / 2;
    setRange(center - nextSpan / 2, center + nextSpan / 2);
  }

  zoomIn.addEventListener("click", zoomInChart);
  zoomOut.addEventListener("click", zoomOutChart);
  zoomReset.addEventListener("click", () => setRange(0, data.values.length - 1));
  focusTest.addEventListener("click", () => setRange(data.trainEnd - 48, data.values.length - 1));
  orderButtons.forEach((button) => button.addEventListener("click", () => {
    state.order = Number(button.dataset.arOrder);
    render();
  }));
  scenarioButtons.forEach((button) => button.addEventListener("click", () => {
    state.scenario = button.dataset.arScenario;
    render();
  }));
  window.addEventListener("theme-changed", render);
  render();
})();
