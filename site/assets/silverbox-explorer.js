(() => {
  const root = document.getElementById("silverbox-explorer");
  const data = window.SILVERBOX_EXPLORER_DATA;
  if (!root || !data) return;

  const responseChart = root.querySelector("#silverbox-response-chart");
  const responseCopy = root.querySelector("#silverbox-response-copy");
  const detail = root.querySelector("#silverbox-model-detail");
  const metrics = root.querySelector("#silverbox-metrics");
  const buttons = [...root.querySelectorAll("[data-silverbox-model]")];
  const zoomButtons = [...root.querySelectorAll("[data-silverbox-zoom]")];
  const panButtons = [...root.querySelectorAll("[data-silverbox-pan]")];
  const windowLabel = root.querySelector("#silverbox-window-label");
  if (!responseChart || !responseCopy || !detail || !metrics || !windowLabel
    || buttons.length === 0 || zoomButtons.length === 0 || panButtons.length === 0) return;

  const SVG_NS = "http://www.w3.org/2000/svg";
  const zoomLengths = {
    1: data.segment.observed.length,
    4: 375,
    8: 188,
    15: 100,
  };
  const state = {
    model: "ar",
    zoomLevel: 1,
    zoomCenter: 433.5,
    zoomStart: 384,
  };
  let dragState = null;

  function colors() {
    const dark = document.documentElement.classList.contains("quarto-dark")
      || document.body.classList.contains("quarto-dark");
    return dark
      ? {
        panel: "#20272b", frame: "#56616a", grid: "#465159", text: "#e5ecef", muted: "#c8d1d6",
        observed: "#88c6e8", prediction: "#f0ad5d",
      }
      : {
        panel: "#ffffff", frame: "#d6e0e4", grid: "#d9e2e6", text: "#34454d", muted: "#5d6a72",
        observed: "#2f7395", prediction: "#dd6b20",
      };
  }

  function svgElement(name, attributes = {}) {
    const element = document.createElementNS(SVG_NS, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
    return element;
  }

  function addText(svg, value, x, y, attributes = {}) {
    const element = svgElement("text", { x, y, ...attributes });
    element.textContent = value;
    svg.append(element);
    return element;
  }

  function linePath(points) {
    return points.map(([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  }

  function finite(values) {
    return values.filter((value) => Number.isFinite(Number(value))).map(Number);
  }

  function format(value, digits = 4) {
    return Number(value).toFixed(digits);
  }

  function setSvgMetadata(svg, { width, height, title, description }) {
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.replaceChildren();
    const titleElement = svgElement("title");
    titleElement.textContent = title;
    svg.append(titleElement);
    const descriptionElement = svgElement("desc");
    descriptionElement.textContent = description;
    svg.append(descriptionElement);
  }

  function tickValues(minimum, maximum) {
    const span = Math.max(maximum - minimum, Number.EPSILON);
    return [minimum, minimum + span / 2, maximum];
  }

  function chartDimensions() {
    const width = 960;
    const height = 264;
    const margin = { left: 68, right: 24, top: 38, bottom: 48 };
    return {
      width,
      height,
      margin,
      right: width - margin.right,
      bottom: height - margin.bottom,
    };
  }

  function drawAxes(svg, { xLeft, xRight, yTop, yBottom, yScale, yValues, colors: palette, yLabel, xScale, time }) {
    yValues.forEach((value) => {
      const y = yScale(value);
      svg.append(svgElement("line", {
        x1: xLeft, x2: xRight, y1: y, y2: y,
        stroke: palette.grid, "stroke-width": 1,
      }));
      addText(svg, format(value, 2), xLeft - 8, y + 4, {
        "text-anchor": "end", "font-size": 10.5, fill: palette.muted,
      });
    });
    svg.append(svgElement("line", {
      x1: xLeft, x2: xLeft, y1: yTop, y2: yBottom,
      stroke: palette.muted, "stroke-width": 1.2,
    }));
    svg.append(svgElement("line", {
      x1: xLeft, x2: xRight, y1: yBottom, y2: yBottom,
      stroke: palette.muted, "stroke-width": 1.2,
    }));
    addText(svg, yLabel, 19, (yTop + yBottom) / 2, {
      transform: `rotate(-90 19 ${(yTop + yBottom) / 2})`, "text-anchor": "middle",
      "font-size": 12, fill: palette.text, "font-weight": 700,
    });

    const indexes = [...new Set([0, Math.round((time.length - 1) / 2), time.length - 1])];
    indexes.forEach((index) => {
      const x = xScale(index);
      addText(svg, `${Number(time[index]).toFixed(2)} s`, x, yBottom + 20, {
        "text-anchor": "middle", "font-size": 10.5, fill: palette.muted,
      });
    });
    addText(svg, "time from start of held-out experiment", (xLeft + xRight) / 2, yBottom + 39, {
      "text-anchor": "middle", "font-size": 11.5, fill: palette.text, "font-weight": 700,
    });
  }

  function drawLine(svg, values, { xScale, yScale, stroke, dash = null, width = 2.1 }) {
    const points = values
      .map((value, index) => (Number.isFinite(Number(value)) ? [xScale(index), yScale(Number(value))] : null))
      .filter(Boolean);
    if (points.length < 2) return;
    const attributes = {
      d: linePath(points), fill: "none", stroke, "stroke-width": width,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    };
    if (dash) attributes["stroke-dasharray"] = dash;
    svg.append(svgElement("path", attributes));
  }

  function drawLegend(svg, { items, x, y, palette }) {
    let currentX = x;
    items.forEach((item) => {
      svg.append(svgElement("line", {
        x1: currentX, x2: currentX + 23, y1: y, y2: y,
        stroke: item.stroke, "stroke-width": 2.5, "stroke-dasharray": item.dash || "",
      }));
      addText(svg, item.label, currentX + 29, y + 4, {
        "font-size": 10.5, fill: palette.muted,
      });
      currentX += item.width;
    });
  }

  function drawSeries(svg, { values, title, description, yLabel, legendItems, start = 0, end = null }) {
    const palette = colors();
    const { width, height, margin, right, bottom } = chartDimensions();
    const sourceTime = data.segment.time
      || data.segment.timeMilliseconds.map((value) => Number(value) / 1000);
    const visibleEnd = end === null ? sourceTime.length : Math.min(end, sourceTime.length);
    const visibleStart = Math.max(0, Math.min(start, visibleEnd - 1));
    const time = sourceTime.slice(visibleStart, visibleEnd);
    const visibleSeries = values.map((series) => ({
      ...series,
      values: series.values.slice(visibleStart, visibleEnd),
    }));
    const chartValues = finite(visibleSeries.flatMap((series) => series.values));
    const lower = Math.min(...chartValues);
    const upper = Math.max(...chartValues);
    const padding = Math.max((upper - lower) * 0.16, 0.035);
    const yMinimum = lower - padding;
    const yMaximum = upper + padding;
    const xScale = (index) => margin.left + (index / Math.max(1, time.length - 1)) * (right - margin.left);
    const yScale = (value) => margin.top + ((yMaximum - value) / (yMaximum - yMinimum)) * (bottom - margin.top);

    setSvgMetadata(svg, { width, height, title, description });
    svg.append(svgElement("rect", {
      x: margin.left, y: margin.top, width: right - margin.left, height: bottom - margin.top,
      fill: palette.panel, stroke: palette.frame, "stroke-width": 1,
    }));
    drawAxes(svg, {
      xLeft: margin.left, xRight: right, yTop: margin.top, yBottom: bottom,
      yScale, yValues: tickValues(yMinimum, yMaximum), colors: palette, yLabel, xScale, time,
    });
    visibleSeries.forEach((series) => drawLine(svg, series.values, {
      xScale, yScale, stroke: series.stroke, dash: series.dash, width: series.width,
    }));
    drawLegend(svg, { items: legendItems, x: right - 260, y: margin.top + 17, palette });
  }

  function selectedModel() {
    return data.metrics[state.model];
  }

  function zoomed() {
    return state.zoomLevel > 1;
  }

  function zoomWindowLength() {
    return zoomLengths[state.zoomLevel];
  }

  function responseRange() {
    if (zoomed()) {
      return { start: state.zoomStart, end: state.zoomStart + zoomWindowLength() };
    }
    return { start: 0, end: data.segment.observed.length };
  }

  function maximumZoomStart() {
    return Math.max(0, data.segment.observed.length - zoomWindowLength());
  }

  function setZoomStart(start) {
    state.zoomStart = Math.max(0, Math.min(maximumZoomStart(), Math.round(start)));
    state.zoomCenter = state.zoomStart + (zoomWindowLength() - 1) / 2;
  }

  function setZoomLevel(level) {
    if (!zoomLengths[level]) return;
    state.zoomLevel = level;
    if (zoomed()) {
      setZoomStart(state.zoomCenter - (zoomWindowLength() - 1) / 2);
    }
  }

  function responseTime(range) {
    const time = data.segment.time
      || data.segment.timeMilliseconds.map((value) => Number(value) / 1000);
    return { start: Number(time[range.start]), end: Number(time[range.end - 1]) };
  }

  function modelExplanation(key) {
    if (key === "ar") {
      return "It recursively reuses only its earlier response forecasts. It cannot respond directly when the known excitation changes.";
    }
    if (key === "arx") {
      return "It receives the changing forcing signal as well as prior responses, but its input-output rule remains linear.";
    }
    return "After a 20-response warmup, it carries a learned nonlinear state from sample to sample. Each step uses the known forcing input and its previous response forecast.";
  }

  function shortUses(model, key) {
    if (model.shortUses) return Array.isArray(model.shortUses) ? model.shortUses.join(" + ") : model.shortUses;
    if (key === "ar") return "past response";
    if (key === "arx") return "past response + forcing";
    return "past response + forcing + state";
  }

  function renderResponse() {
    const model = selectedModel();
    const range = responseRange();
    const time = responseTime(range);
    const duration = time.end - time.start;
    const isZoomed = zoomed();
    responseChart.classList.toggle("is-pannable", isZoomed);
    responseCopy.textContent = isZoomed
      ? `Zoom ${state.zoomLevel}×: ${duration.toFixed(2)} s (${time.start.toFixed(2)}–${time.end.toFixed(2)} s). Drag the chart left or right to pan, then switch models to compare the same interval.`
      : "Full held-out record. Choose a zoom level to inspect a short interval and compare the forecasts.";
    drawSeries(responseChart, {
      title: `${model.label} recursive forecast on the held-out Silverbox signal`,
      description: `Observed output is blue. The dashed orange curve is the selected ${model.label} recursive forecast. ${isZoomed ? `This ${state.zoomLevel}× view covers ${duration.toFixed(2)} seconds of the held-out record and can be panned by dragging.` : ""}`,
      yLabel: "response yₜ",
      legendItems: [
        { label: "observed yₜ", stroke: colors().observed, width: 116 },
        { label: `${model.label} forecast`, stroke: colors().prediction, dash: "6 3", width: 130 },
      ],
      values: [
        { values: data.segment.observed, stroke: colors().observed, width: 2.05 },
        { values: data.segment.predictions[state.model], stroke: colors().prediction, dash: "6 3", width: 2.1 },
      ],
      start: range.start,
      end: range.end,
    });
  }

  function renderDetail() {
    const model = selectedModel();
    const uses = Array.isArray(model.uses) ? model.uses : [model.uses];
    detail.replaceChildren();
    const heading = document.createElement("p");
    heading.className = "silverbox-detail-kicker";
    heading.textContent = `Selected: ${model.label}`;
    detail.append(heading);

    const value = document.createElement("strong");
    value.className = "silverbox-detail-rmse";
    value.textContent = `Held-out recursive RMSE = ${format(model.rmse)}`;
    detail.append(value);

    const explanation = document.createElement("p");
    explanation.className = "silverbox-detail-copy";
    explanation.textContent = model.explanation || modelExplanation(state.model);
    detail.append(explanation);

    const list = document.createElement("ul");
    list.className = "silverbox-information-list";
    uses.forEach((item) => {
      const listItem = document.createElement("li");
      listItem.textContent = item;
      list.append(listItem);
    });
    detail.append(list);
  }

  function renderMetrics() {
    metrics.replaceChildren();
    const table = document.createElement("table");
    table.className = "silverbox-metrics-table";
    const caption = document.createElement("caption");
    caption.textContent = "Same held-out recursive forecasting task";
    table.append(caption);
    const head = document.createElement("thead");
    const headerRow = document.createElement("tr");
    ["Model", "Information available", "Recursive RMSE"].forEach((label) => {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      headerRow.append(cell);
    });
    head.append(headerRow);
    table.append(head);
    const body = document.createElement("tbody");
    ["ar", "arx", "rnn"].forEach((key) => {
      const model = data.metrics[key];
      if (!model) return;
      const row = document.createElement("tr");
      if (key === state.model) row.classList.add("is-selected");
      const label = document.createElement("th");
      label.scope = "row";
      label.textContent = model.label;
      const signal = document.createElement("td");
      signal.textContent = shortUses(model, key);
      const rmse = document.createElement("td");
      rmse.textContent = format(model.rmse);
      row.append(label, signal, rmse);
      body.append(row);
    });
    table.append(body);
    metrics.append(table);
  }

  function renderButtons() {
    buttons.forEach((button) => {
      const active = button.dataset.silverboxModel === state.model;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }

  function renderZoomButtons() {
    zoomButtons.forEach((button) => {
      const active = Number(button.dataset.silverboxZoom) === state.zoomLevel;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }

  function renderPanControls() {
    const range = responseRange();
    const time = responseTime(range);
    const isZoomed = zoomed();
    panButtons.forEach((button) => {
      const direction = button.dataset.silverboxPan;
      const atBeginning = state.zoomStart <= 0;
      const atEnd = state.zoomStart >= maximumZoomStart();
      button.disabled = !isZoomed
        || (direction === "earlier" && atBeginning)
        || (direction === "later" && atEnd);
    });
    windowLabel.textContent = isZoomed
      ? `${time.start.toFixed(2)}–${time.end.toFixed(2)} s`
      : "Choose a zoom level to pan";
  }

  function render() {
    renderButtons();
    renderZoomButtons();
    renderPanControls();
    renderResponse();
    renderDetail();
    renderMetrics();
  }

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const model = button.dataset.silverboxModel;
      if (!data.metrics[model]) return;
      state.model = model;
      render();
    });
  });

  zoomButtons.forEach((button) => {
    button.addEventListener("click", () => {
      setZoomLevel(Number(button.dataset.silverboxZoom));
      render();
    });
  });

  panButtons.forEach((button) => {
    button.addEventListener("click", () => {
      if (!zoomed()) return;
      const direction = button.dataset.silverboxPan;
      const step = Math.max(1, Math.round(zoomWindowLength() / 2));
      setZoomStart(state.zoomStart + (direction === "earlier" ? -step : step));
      render();
    });
  });

  responseChart.addEventListener("pointerdown", (event) => {
    if (!zoomed() || event.button !== 0) return;
    const rectangle = responseChart.getBoundingClientRect();
    const dimensions = chartDimensions();
    const svgX = (event.clientX - rectangle.left) / rectangle.width * dimensions.width;
    const svgY = (event.clientY - rectangle.top) / rectangle.height * dimensions.height;
    if (svgX < dimensions.margin.left || svgX > dimensions.right
      || svgY < dimensions.margin.top || svgY > dimensions.bottom) return;
    const plotWidth = rectangle.width * (dimensions.right - dimensions.margin.left) / dimensions.width;
    dragState = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      zoomStart: state.zoomStart,
      plotWidth,
    };
    responseChart.setPointerCapture?.(event.pointerId);
    responseChart.classList.add("is-panning");
    event.preventDefault();
  });

  responseChart.addEventListener("pointermove", (event) => {
    if (!dragState || event.pointerId !== dragState.pointerId) return;
    const samplesMoved = Math.round((dragState.clientX - event.clientX) / dragState.plotWidth * zoomWindowLength());
    const nextStart = dragState.zoomStart + samplesMoved;
    if (nextStart === state.zoomStart) return;
    setZoomStart(nextStart);
    render();
  });

  function endChartPan(event) {
    if (!dragState || event.pointerId !== dragState.pointerId) return;
    responseChart.releasePointerCapture?.(event.pointerId);
    responseChart.classList.remove("is-panning");
    dragState = null;
  }

  responseChart.addEventListener("pointerup", endChartPan);
  responseChart.addEventListener("pointercancel", endChartPan);

  render();
})();
