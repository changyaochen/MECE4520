(() => {
  // Same centroids as np.random.seed(42); np.random.normal(size=(3, 2)).
  const INITIAL_CENTERS = [
    [0.4967141530112327, -0.13826430117118466],
    [0.6476885381006925, 1.5230298564080254],
    [-0.23415337472333597, -0.23413695694918055],
  ];
  const COLORS = ["#237cb0", "#dc8d24", "#a65383"];

  function distanceSquared(point, center) {
    return (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2;
  }

  function cost(points, labels, centers) {
    return points.reduce((sum, point, i) => sum + distanceSquared(point, centers[labels[i]]), 0);
  }

  function buildTrace(points, initialCenters = INITIAL_CENTERS) {
    let centers = initialCenters.map((center) => center.slice());
    const trace = [{ phase: "initial", iteration: 0, centers, labels: null, objective: null, counts: null }];
    for (let iteration = 1; iteration <= 100; iteration += 1) {
      const labels = points.map((point) => {
        let best = 0;
        for (let i = 1; i < centers.length; i += 1) {
          if (distanceSquared(point, centers[i]) < distanceSquared(point, centers[best])) best = i;
        }
        return best;
      });
      const counts = centers.map(() => 0);
      const sums = centers.map(() => [0, 0]);
      points.forEach((point, i) => {
        const label = labels[i];
        counts[label] += 1;
        sums[label][0] += point[0];
        sums[label][1] += point[1];
      });
      trace.push({ phase: "assign", iteration, centers, labels, counts, objective: cost(points, labels, centers) });

      // If a cluster is empty, keep its previous center; no mean exists for an empty group.
      const nextCenters = centers.map((center, i) => counts[i]
        ? [sums[i][0] / counts[i], sums[i][1] / counts[i]] : center.slice());
      const converged = centers.every((center, i) => distanceSquared(center, nextCenters[i]) < 1e-20);
      trace.push({ phase: "update", iteration, centers: nextCenters, previousCenters: centers,
        labels, counts, objective: cost(points, labels, nextCenters), converged });
      centers = nextCenters;
      if (converged) return trace;
    }
    trace[trace.length - 1].limitReached = true;
    return trace;
  }

  // Allow a standalone test to compare the step sequence against another implementation.
  if (typeof module !== "undefined" && module.exports) module.exports = { buildTrace, INITIAL_CENTERS };
  if (typeof document === "undefined") return;
  const root = document.getElementById("kmeans-explorer");
  if (!root) return;
  const get = (id) => document.getElementById(`kmeans-${id}`);
  const plot = get("plot");
  const previous = get("previous");
  const next = get("next");
  const play = get("play");
  const reset = get("reset");
  const progress = get("progress");
  const action = get("action");
  const explanation = get("explanation");
  const iteration = get("iteration");
  const objective = get("objective");
  const counts = get("counts");
  let points = [];
  let trace = [];
  let step = 0;
  let timer = null;

  const NS = "http://www.w3.org/2000/svg";
  function svg(name, attributes = {}) {
    const element = document.createElementNS(NS, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
    return element;
  }
  function text(parent, value, x, y, attributes = {}) {
    const element = svg("text", { x, y, ...attributes });
    element.textContent = value;
    parent.append(element);
  }
  const area = { left: 68, top: 30, size: 520, min: -13, max: 13 };
  const toX = (x) => area.left + (x - area.min) / (area.max - area.min) * area.size;
  const toY = (y) => area.top + (area.max - y) / (area.max - area.min) * area.size;

  function draw(state) {
    plot.replaceChildren();
    const title = svg("title");
    title.textContent = `Iteration ${state.iteration}: ${state.phase}`;
    plot.append(title);
    const defs = svg("defs");
    const marker = svg("marker", { id: "kmeans-arrow-head", viewBox: "0 0 10 10", refX: 9, refY: 5,
      markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" });
    marker.append(svg("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#536b76" }));
    defs.append(marker);
    plot.append(defs);
    [-10, -5, 0, 5, 10].forEach((tick) => {
      const px = toX(tick);
      const py = toY(tick);
      const css = tick === 0 ? "kmeans-zero" : "kmeans-grid";
      plot.append(svg("line", { x1: px, x2: px, y1: area.top, y2: area.top + area.size, class: css }));
      plot.append(svg("line", { x1: area.left, x2: area.left + area.size, y1: py, y2: py, class: css }));
      text(plot, tick, px, area.top + area.size + 22, { class: "kmeans-tick", "text-anchor": "middle" });
      text(plot, tick, area.left - 12, py + 4, { class: "kmeans-tick", "text-anchor": "end" });
    });
    plot.append(svg("line", { x1: area.left, x2: area.left + area.size, y1: area.top + area.size,
      y2: area.top + area.size, class: "kmeans-axis" }));
    plot.append(svg("line", { x1: area.left, x2: area.left, y1: area.top, y2: area.top + area.size,
      class: "kmeans-axis" }));
    text(plot, "Feature 1", area.left + area.size / 2, 595,
      { class: "kmeans-axis-label", "text-anchor": "middle" });
    text(plot, "Feature 2", 20, area.top + area.size / 2,
      { class: "kmeans-axis-label", "text-anchor": "middle", transform: `rotate(-90 20 ${area.top + area.size / 2})` });

    points.forEach((point, i) => {
      const fill = state.labels ? COLORS[state.labels[i]] : "#738c9a";
      plot.append(svg("circle", { cx: toX(point[0]), cy: toY(point[1]), r: 4,
        class: "kmeans-point", fill }));
    });
    if (state.previousCenters) {
      state.centers.forEach((center, i) => {
        const old = state.previousCenters[i];
        if (distanceSquared(center, old) > 0.01) {
          plot.append(svg("line", { x1: toX(old[0]), y1: toY(old[1]), x2: toX(center[0]), y2: toY(center[1]),
            class: "kmeans-movement", "marker-end": "url(#kmeans-arrow-head)" }));
        }
      });
    }
    state.centers.forEach((center, i) => {
      const x = toX(center[0]);
      const y = toY(center[1]);
      [[-9, -9, 9, 9], [-9, 9, 9, -9]].forEach(([dx1, dy1, dx2, dy2]) => {
        const coords = { x1: x + dx1, y1: y + dy1, x2: x + dx2, y2: y + dy2, "stroke-linecap": "round" };
        plot.append(svg("line", { ...coords, stroke: "white", "stroke-width": 8 }));
        plot.append(svg("line", { ...coords, stroke: COLORS[i], class: "kmeans-center" }));
      });
    });
  }

  function render() {
    const state = trace[step];
    draw(state);
    previous.disabled = step === 0;
    next.disabled = step === trace.length - 1;
    play.disabled = step === trace.length - 1 && !timer;
    progress.textContent = `Step ${step} of ${trace.length - 1}`;
    iteration.textContent = state.iteration;
    objective.textContent = state.objective === null ? "—" : state.objective.toLocaleString(undefined, { maximumFractionDigits: 1 });
    counts.replaceChildren();
    COLORS.forEach((color, i) => {
      const row = document.createElement("div");
      row.className = "kmeans-count";
      const label = document.createElement("span");
      label.className = "kmeans-count-label";
      const dot = document.createElement("i");
      dot.className = "kmeans-count-dot";
      dot.style.backgroundColor = color;
      label.append(dot, document.createTextNode(`Cluster ${i + 1}`));
      const count = document.createElement("strong");
      count.textContent = state.counts ? state.counts[i] : "—";
      row.append(label, count);
      counts.append(row);
    });
    if (state.phase === "initial") {
      action.textContent = "Start: choose three initial centroids";
      explanation.textContent = "The points are not assigned yet. Next, assign each point to its closest centroid.";
    } else if (state.phase === "assign") {
      action.textContent = `Iteration ${state.iteration}: assign points`;
      explanation.textContent = "Each point is assigned to the nearest centroid. The centroid positions have not changed yet.";
    } else if (state.converged) {
      action.textContent = `Iteration ${state.iteration}: converged`;
      explanation.textContent = "The centroid positions no longer change, so the algorithm stops.";
    } else if (state.limitReached) {
      action.textContent = `Iteration ${state.iteration}: update centroids`;
      explanation.textContent = "The demonstration reached its 100-iteration limit before the centroids stopped moving.";
    } else {
      action.textContent = `Iteration ${state.iteration}: update centroids`;
      explanation.textContent = state.counts.includes(0)
        ? "Each centroid moves to the mean of its assigned points. An empty cluster keeps its previous centroid."
        : "Each centroid moves to the mean of its assigned points. Next, assign the points again.";
    }
  }
  function stop() {
    if (timer !== null) window.clearInterval(timer);
    timer = null;
    play.textContent = "Play";
  }
  function stepForward() {
    if (step < trace.length - 1) step += 1;
    if (step === trace.length - 1) stop();
    render();
  }
  previous.addEventListener("click", () => { stop(); if (step > 0) step -= 1; render(); });
  next.addEventListener("click", () => { stop(); stepForward(); });
  reset.addEventListener("click", () => { stop(); step = 0; render(); });
  play.addEventListener("click", () => {
    if (timer !== null) { stop(); render(); return; }
    play.textContent = "Pause";
    timer = window.setInterval(stepForward, 950);
    render();
  });
  [previous, next, play, reset].forEach((button) => { button.disabled = true; });
  fetch("../data/kmeans-three-blobs.csv")
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.text();
    })
    .then((csv) => {
      points = csv.trim().split(/\r?\n/).slice(1).map((line) => line.split(",").map(Number));
      if (!points.length || points.some((point) => point.length !== 2 || point.some((x) => !Number.isFinite(x)))) {
        throw new Error("Invalid point data");
      }
      trace = buildTrace(points);
      reset.disabled = false;
      render();
    })
    .catch(() => { progress.textContent = "Could not load the data. Reload the page to try again.";
      action.textContent = "Data unavailable"; });
})();
