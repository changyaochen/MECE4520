(() => {
  const root = document.getElementById("pca-explorer");
  if (!root) return;

  const angleSlider = root.querySelector("#pca-angle");
  const angleOutput = root.querySelector("#pca-angle-value");
  const playButton = root.querySelector("#pca-play");
  const resetButton = root.querySelector("#pca-reset");
  const status = root.querySelector("#pca-status");
  const scatter = root.querySelector("#pca-scatter");
  const xHistogram = root.querySelector("#pca-x-histogram");
  const yHistogram = root.querySelector("#pca-y-histogram");
  const projectionHistogram = root.querySelector("#pca-projection-histogram");
  const varianceOutput = root.querySelector("#pca-variance");
  const varianceShareOutput = root.querySelector("#pca-variance-share");
  const lambdaOneOutput = root.querySelector("#pca-lambda-one");
  const lambdaTwoOutput = root.querySelector("#pca-lambda-two");
  const orientationNote = root.querySelector("#pca-orientation-note");

  if ([angleSlider, angleOutput, playButton, resetButton, status, scatter,
    xHistogram, yHistogram, projectionHistogram, varianceOutput,
    varianceShareOutput, lambdaOneOutput, lambdaTwoOutput, orientationNote]
    .some((element) => !element)) return;

  const SVG_NS = "http://www.w3.org/2000/svg";
  const DOMAIN = { min: -3.5, max: 3.5 };
  const PLOT = { left: 58, top: 28, size: 420 };
  const MARGINAL_BINS = 12;
  const PROJECTION_BINS = 20;
  const state = { points: [], angle: 0, playing: false, frame: null, lastFrame: 0 };

  function svgElement(name, attributes = {}) {
    const element = document.createElementNS(SVG_NS, name);
    Object.entries(attributes).forEach(([key, value]) => {
      element.setAttribute(key, String(value));
    });
    return element;
  }

  function addText(svg, text, x, y, attributes = {}) {
    const element = svgElement("text", { x, y, ...attributes });
    element.textContent = text;
    svg.append(element);
    return element;
  }

  function format(value, digits = 2) {
    const rounded = Number(value.toFixed(digits));
    return Math.abs(rounded) < 10 ** (-digits) ? "0" : rounded.toFixed(digits);
  }

  function normalizeDegrees(degrees) {
    return ((degrees % 360) + 360) % 360;
  }

  function normalizeRadians(radians) {
    return ((radians % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  }

  function xPixel(value) {
    return PLOT.left + ((value - DOMAIN.min) / (DOMAIN.max - DOMAIN.min)) * PLOT.size;
  }

  function yPixel(value) {
    return PLOT.top + ((DOMAIN.max - value) / (DOMAIN.max - DOMAIN.min)) * PLOT.size;
  }

  function histogram(values, min, max, binCount) {
    const counts = Array(binCount).fill(0);
    const width = (max - min) / binCount;
    values.forEach((value) => {
      const index = Math.min(binCount - 1, Math.max(0, Math.floor((value - min) / width)));
      counts[index] += 1;
    });
    return { counts, width };
  }

  function computePca(points) {
    const meanX = points.reduce((sum, point) => sum + point.x, 0) / points.length;
    const meanY = points.reduce((sum, point) => sum + point.y, 0) / points.length;
    let covarianceXX = 0;
    let covarianceXY = 0;
    let covarianceYY = 0;
    const centered = points.map((point) => ({ x: point.x - meanX, y: point.y - meanY }));

    centered.forEach((point) => {
      covarianceXX += point.x * point.x;
      covarianceXY += point.x * point.y;
      covarianceYY += point.y * point.y;
    });

    const divisor = points.length - 1;
    covarianceXX /= divisor;
    covarianceXY /= divisor;
    covarianceYY /= divisor;

    const root = Math.sqrt(
      ((covarianceXX - covarianceYY) ** 2) + (4 * covarianceXY * covarianceXY),
    );
    const lambdaOne = (covarianceXX + covarianceYY + root) / 2;
    const lambdaTwo = (covarianceXX + covarianceYY - root) / 2;
    const pc1Angle = 0.5 * Math.atan2(2 * covarianceXY, covarianceXX - covarianceYY);

    return {
      meanX,
      meanY,
      covarianceXX,
      covarianceXY,
      covarianceYY,
      lambdaOne,
      lambdaTwo,
      pc1Angle: normalizeRadians(pc1Angle),
      centered,
      totalVariance: lambdaOne + lambdaTwo,
    };
  }

  function projectionValues(angleRadians) {
    const cosine = Math.cos(angleRadians);
    const sine = Math.sin(angleRadians);
    return state.pca.centered.map((point) => (point.x * cosine) + (point.y * sine));
  }

  function projectionVariance(angleRadians) {
    const { covarianceXX: xx, covarianceXY: xy, covarianceYY: yy } = state.pca;
    const cosine = Math.cos(angleRadians);
    const sine = Math.sin(angleRadians);
    return (cosine * cosine * xx) + (2 * cosine * sine * xy) + (sine * sine * yy);
  }

  function angularDistanceAsAxis(first, second) {
    const difference = (((first - second + (Math.PI / 2)) % Math.PI) + Math.PI) % Math.PI;
    return Math.abs(difference - (Math.PI / 2));
  }

  function lineEndpoints(angleRadians) {
    const cosine = Math.cos(angleRadians);
    const sine = Math.sin(angleRadians);
    let lower = -Infinity;
    let upper = Infinity;
    [
      [state.pca.meanX, cosine],
      [state.pca.meanY, sine],
    ].forEach(([center, direction]) => {
      if (Math.abs(direction) < 1e-10) return;
      const first = (DOMAIN.min - center) / direction;
      const second = (DOMAIN.max - center) / direction;
      lower = Math.max(lower, Math.min(first, second));
      upper = Math.min(upper, Math.max(first, second));
    });
    return [
      { x: state.pca.meanX + lower * cosine, y: state.pca.meanY + lower * sine },
      { x: state.pca.meanX + upper * cosine, y: state.pca.meanY + upper * sine },
    ];
  }

  function drawScatter() {
    scatter.replaceChildren();
    const title = svgElement("title");
    title.textContent = "The original point cloud, PC1 reference direction, and current orthogonal projections";
    scatter.append(title);

    const ticks = [-3, -2, -1, 0, 1, 2, 3];
    ticks.forEach((tick) => {
      const x = xPixel(tick);
      const y = yPixel(tick);
      scatter.append(svgElement("line", {
        x1: x, x2: x, y1: PLOT.top, y2: PLOT.top + PLOT.size,
        class: tick === 0 ? "pca-zero-axis" : "pca-gridline",
      }));
      scatter.append(svgElement("line", {
        x1: PLOT.left, x2: PLOT.left + PLOT.size, y1: y, y2: y,
        class: tick === 0 ? "pca-zero-axis" : "pca-gridline",
      }));
      scatter.append(svgElement("line", {
        x1: x, x2: x, y1: PLOT.top + PLOT.size, y2: PLOT.top + PLOT.size + 5,
        class: "pca-axis-line",
      }));
      scatter.append(svgElement("line", {
        x1: PLOT.left - 5, x2: PLOT.left, y1: y, y2: y,
        class: "pca-axis-line",
      }));
      addText(scatter, String(tick), x, PLOT.top + PLOT.size + 21, {
        class: "pca-chart-tick", "text-anchor": "middle",
      });
      addText(scatter, String(tick), PLOT.left - 10, y + 4, {
        class: "pca-chart-tick", "text-anchor": "end",
      });
    });

    scatter.append(svgElement("line", {
      x1: PLOT.left,
      x2: PLOT.left + PLOT.size,
      y1: PLOT.top + PLOT.size,
      y2: PLOT.top + PLOT.size,
      class: "pca-axis-line",
    }));
    scatter.append(svgElement("line", {
      x1: PLOT.left,
      x2: PLOT.left,
      y1: PLOT.top,
      y2: PLOT.top + PLOT.size,
      class: "pca-axis-line",
    }));
    addText(scatter, "x", PLOT.left + (PLOT.size / 2), 515, {
      class: "pca-chart-axis-label", "text-anchor": "middle",
    });
    addText(scatter, "y", 17, PLOT.top + (PLOT.size / 2), {
      class: "pca-chart-axis-label",
      "text-anchor": "middle",
      transform: `rotate(-90 17 ${PLOT.top + (PLOT.size / 2)})`,
    });

    const pc1Endpoints = lineEndpoints(state.pca.pc1Angle);
    scatter.append(svgElement("line", {
      x1: xPixel(pc1Endpoints[0].x),
      y1: yPixel(pc1Endpoints[0].y),
      x2: xPixel(pc1Endpoints[1].x),
      y2: yPixel(pc1Endpoints[1].y),
      class: "pca-pc1-line",
    }));
    const pc1LabelPoint = pc1Endpoints[1];
    addText(scatter, "PC1", xPixel(pc1LabelPoint.x) - 8, yPixel(pc1LabelPoint.y) + 18, {
      class: "pca-pc1-label", "text-anchor": "end",
    });

    const angleRadians = state.angle * Math.PI / 180;
    const nearPc1 = angularDistanceAsAxis(angleRadians, state.pca.pc1Angle) < (Math.PI / 60);
    const nearPc2 = angularDistanceAsAxis(angleRadians, state.pca.pc1Angle + (Math.PI / 2)) < (Math.PI / 60);
    const currentEndpoints = lineEndpoints(angleRadians);

    state.points.forEach((point, index) => {
      const centered = state.pca.centered[index];
      const projection = (centered.x * Math.cos(angleRadians)) + (centered.y * Math.sin(angleRadians));
      const projectedX = state.pca.meanX + projection * Math.cos(angleRadians);
      const projectedY = state.pca.meanY + projection * Math.sin(angleRadians);
      scatter.append(svgElement("line", {
        x1: xPixel(point.x),
        y1: yPixel(point.y),
        x2: xPixel(projectedX),
        y2: yPixel(projectedY),
        class: "pca-projection-segment",
      }));
    });

    scatter.append(svgElement("line", {
      x1: xPixel(currentEndpoints[0].x),
      y1: yPixel(currentEndpoints[0].y),
      x2: xPixel(currentEndpoints[1].x),
      y2: yPixel(currentEndpoints[1].y),
      class: nearPc1 ? "pca-current-direction pca-current-direction-pc1" : "pca-current-direction",
    }));

    state.points.forEach((point, index) => {
      const centered = state.pca.centered[index];
      const projection = (centered.x * Math.cos(angleRadians)) + (centered.y * Math.sin(angleRadians));
      const projectedX = state.pca.meanX + projection * Math.cos(angleRadians);
      const projectedY = state.pca.meanY + projection * Math.sin(angleRadians);
      scatter.append(svgElement("circle", {
        cx: xPixel(projectedX), cy: yPixel(projectedY), r: 2.5,
        class: "pca-projected-point",
      }));
    });

    state.points.forEach((point) => {
      scatter.append(svgElement("circle", {
        cx: xPixel(point.x), cy: yPixel(point.y), r: 3.1,
        class: "pca-original-point",
      }));
    });

    scatter.append(svgElement("circle", {
      cx: xPixel(state.pca.meanX), cy: yPixel(state.pca.meanY), r: 4.5,
      class: "pca-data-mean",
    }));
    addText(scatter, "mean", xPixel(state.pca.meanX) + 8, yPixel(state.pca.meanY) - 7, {
      class: "pca-mean-label",
    });
  }

  function drawMarginals() {
    const xValues = state.points.map((point) => point.x);
    const yValues = state.points.map((point) => point.y);
    const xHistogramData = histogram(xValues, DOMAIN.min, DOMAIN.max, MARGINAL_BINS);
    const yHistogramData = histogram(yValues, DOMAIN.min, DOMAIN.max, MARGINAL_BINS);
    const xMax = Math.max(...xHistogramData.counts);
    const yMax = Math.max(...yHistogramData.counts);
    const xBase = 88;
    const xTop = 12;
    const xBinPixels = PLOT.size / MARGINAL_BINS;

    xHistogram.replaceChildren();
    addText(xHistogram, "Original x", PLOT.left, 12, { class: "pca-marginal-label" });
    xHistogramData.counts.forEach((count, index) => {
      const height = (count / xMax) * (xBase - xTop);
      const value = DOMAIN.min + (index * xHistogramData.width);
      xHistogram.append(svgElement("rect", {
        x: xPixel(value) + 1,
        y: xBase - height,
        width: Math.max(1, xBinPixels - 2),
        height,
        class: "pca-marginal-bar",
      }));
    });
    xHistogram.append(svgElement("line", {
      x1: PLOT.left, x2: PLOT.left + PLOT.size, y1: xBase, y2: xBase,
      class: "pca-marginal-baseline",
    }));
    addText(xHistogram, "frequency", PLOT.left + PLOT.size - 2, 16, {
      class: "pca-marginal-note", "text-anchor": "end",
    });

    yHistogram.replaceChildren();
    addText(yHistogram, "frequency", 142, 16, {
      class: "pca-marginal-note", "text-anchor": "end",
    });
    const yBinPixels = PLOT.size / MARGINAL_BINS;
    yHistogramData.counts.forEach((count, index) => {
      const value = DOMAIN.min + ((index + 0.5) * yHistogramData.width);
      const width = (count / yMax) * 105;
      yHistogram.append(svgElement("rect", {
        x: 12,
        y: yPixel(value) - (yBinPixels * 0.38),
        width,
        height: yBinPixels * 0.76,
        class: "pca-marginal-bar",
      }));
    });
    yHistogram.append(svgElement("line", {
      x1: 12, x2: 120, y1: PLOT.top + PLOT.size, y2: PLOT.top + PLOT.size,
      class: "pca-marginal-baseline",
    }));
  }

  function projectionHistogramSettings() {
    const largestRadius = Math.max(...state.pca.centered.map((point) => Math.hypot(point.x, point.y)));
    const limit = Math.ceil(largestRadius * 2) / 2;
    let tallestBar = 0;
    for (let degrees = 0; degrees < 360; degrees += 2) {
      const values = projectionValues(degrees * Math.PI / 180);
      tallestBar = Math.max(tallestBar, ...histogram(values, -limit, limit, PROJECTION_BINS).counts);
    }
    return {
      limit,
      countMax: Math.max(4, Math.ceil((tallestBar * 1.16) / 2) * 2),
    };
  }

  function drawProjectionHistogram(angleRadians) {
    const values = projectionValues(angleRadians);
    const { counts, width: binWidth } = histogram(
      values,
      -state.histogram.limit,
      state.histogram.limit,
      PROJECTION_BINS,
    );
    const width = 430;
    const height = 440;
    const plot = { left: 54, right: 412, top: 34, bottom: 365 };
    const plotWidth = plot.right - plot.left;
    const plotHeight = plot.bottom - plot.top;
    const x = (value) => plot.left + ((value + state.histogram.limit) / (2 * state.histogram.limit)) * plotWidth;
    const y = (count) => plot.bottom - (count / state.histogram.countMax) * plotHeight;
    projectionHistogram.replaceChildren();
    const title = svgElement("title");
    title.textContent = "Histogram of all values projected onto the current direction, with fixed axes";
    projectionHistogram.append(title);

    [0, Math.ceil(state.histogram.countMax / 2), state.histogram.countMax].forEach((tick) => {
      projectionHistogram.append(svgElement("line", {
        x1: plot.left, x2: plot.right, y1: y(tick), y2: y(tick),
        class: tick === 0 ? "pca-hist-axis" : "pca-hist-gridline",
      }));
      addText(projectionHistogram, String(tick), plot.left - 9, y(tick) + 4, {
        class: "pca-chart-tick", "text-anchor": "end",
      });
    });

    const angle = angleRadians;
    const nearPc1 = angularDistanceAsAxis(angle, state.pca.pc1Angle) < (Math.PI / 60);
    const fillClass = nearPc1 ? "pca-projection-bar pca-projection-bar-pc1" : "pca-projection-bar";
    const binPixels = plotWidth / PROJECTION_BINS;
    counts.forEach((count, index) => {
      const leftValue = -state.histogram.limit + (index * binWidth);
      projectionHistogram.append(svgElement("rect", {
        x: x(leftValue) + 1,
        y: y(count),
        width: Math.max(1, binPixels - 2),
        height: Math.max(0, plot.bottom - y(count)),
        class: fillClass,
      }));
    });

    projectionHistogram.append(svgElement("line", {
      x1: x(0), x2: x(0), y1: plot.top, y2: plot.bottom,
      class: "pca-hist-zero",
    }));
    projectionHistogram.append(svgElement("line", {
      x1: plot.left, x2: plot.right, y1: plot.bottom, y2: plot.bottom,
      class: "pca-hist-axis",
    }));
    [-state.histogram.limit, 0, state.histogram.limit].forEach((tick) => {
      addText(projectionHistogram, format(tick, 1), x(tick), plot.bottom + 20, {
        class: "pca-chart-tick", "text-anchor": "middle",
      });
    });
    addText(projectionHistogram, "centered projected value, z", (plot.left + plot.right) / 2, 414, {
      class: "pca-chart-axis-label", "text-anchor": "middle",
    });
    addText(projectionHistogram, "count", 15, (plot.top + plot.bottom) / 2, {
      class: "pca-chart-axis-label",
      "text-anchor": "middle",
      transform: `rotate(-90 15 ${(plot.top + plot.bottom) / 2})`,
    });
  }

  function render() {
    const angleRadians = state.angle * Math.PI / 180;
    const variance = projectionVariance(angleRadians);
    const share = variance / state.pca.totalVariance;
    const nearPc1 = angularDistanceAsAxis(angleRadians, state.pca.pc1Angle) < (Math.PI / 60);
    const nearPc2 = angularDistanceAsAxis(angleRadians, state.pca.pc1Angle + (Math.PI / 2)) < (Math.PI / 60);

    angleSlider.value = String(Math.round(state.angle));
    angleOutput.textContent = `${Math.round(state.angle)}°`;
    varianceOutput.textContent = format(variance, 3);
    varianceShareOutput.textContent = `${(share * 100).toFixed(1)}%`;
    lambdaOneOutput.textContent = format(state.pca.lambdaOne, 3);
    lambdaTwoOutput.textContent = format(state.pca.lambdaTwo, 3);
    status.textContent = `n = ${state.points.length} observations · sample covariance (ddof = 1)`;

    if (nearPc1) {
      orientationNote.textContent = "Near PC1: the projected spread is at its maximum.";
      root.classList.add("pca-at-pc1");
    } else if (nearPc2) {
      orientationNote.textContent = "Near PC2: the projected spread is at its minimum.";
      root.classList.remove("pca-at-pc1");
    } else {
      orientationNote.textContent = "Keep rotating: the variance changes with the direction.";
      root.classList.remove("pca-at-pc1");
    }

    playButton.textContent = state.playing ? "Pause rotation" : "Play rotation";
    playButton.setAttribute("aria-pressed", String(state.playing));
    resetButton.disabled = state.playing;
    drawScatter();
    drawProjectionHistogram(angleRadians);
  }

  function stopAnimation() {
    state.playing = false;
    state.frame = null;
    state.lastFrame = 0;
  }

  function animate(timestamp) {
    if (!state.playing) return;
    if (timestamp - state.lastFrame >= 1000 / 30) {
      state.angle = normalizeDegrees(state.angle + 1);
      state.lastFrame = timestamp;
      render();
    }
    state.frame = window.requestAnimationFrame(animate);
  }

  function toggleAnimation() {
    if (state.playing) {
      window.cancelAnimationFrame(state.frame);
      stopAnimation();
      render();
      return;
    }
    state.playing = true;
    state.lastFrame = 0;
    render();
    state.frame = window.requestAnimationFrame(animate);
  }

  function parseCsv(text) {
    return text.trim().split(/\r?\n/).slice(1).map((line) => {
      const [x, y, pixelX, pixelY] = line.split(",").map(Number);
      return { x, y, pixelX, pixelY };
    }).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  }

  function start(points) {
    state.points = points;
    state.pca = computePca(points);
    state.histogram = projectionHistogramSettings();
    drawMarginals();
    angleSlider.addEventListener("input", () => {
      if (state.playing) {
        window.cancelAnimationFrame(state.frame);
        stopAnimation();
      }
      state.angle = Number(angleSlider.value);
      render();
    });
    playButton.addEventListener("click", toggleAnimation);
    resetButton.addEventListener("click", () => {
      state.angle = normalizeDegrees(state.pca.pc1Angle * 180 / Math.PI);
      render();
    });

    // Show the eigenvalues once on load; render() refreshes them with each angle.
    render();
  }

  fetch("../data/pca-demo-points.csv")
    .then((response) => {
      if (!response.ok) throw new Error(`Could not load the point data (${response.status}).`);
      return response.text();
    })
    .then(parseCsv)
    .then((points) => {
      if (points.length < 3) throw new Error("The point data needs at least three observations.");
      start(points);
    })
    .catch((error) => {
      status.textContent = "The PCA point cloud could not be loaded.";
      orientationNote.textContent = error.message;
      playButton.disabled = true;
      resetButton.disabled = true;
    });
})();
