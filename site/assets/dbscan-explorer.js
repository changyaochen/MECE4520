(() => {
  // Record a deterministic step for each neighborhood inspection. Labels of -1
  // mean unassigned; visited unassigned points are provisional noise.
  function buildTrace(points, epsilon, minSamples) {
    const neighbors = points.map((p) => points.flatMap((q, index) =>
      Math.hypot(p.x - q.x, p.y - q.y) <= epsilon + 1e-12 ? [index] : []));
    const visited = points.map(() => false);
    const labels = points.map(() => -1);
    const core = points.map(() => null);
    const trace = [];
    let clusterCount = 0;

    function snapshot(active, kind, message, queueSize = 0) {
      trace.push({
        active, kind, message, queueSize, clusterCount,
        neighbors: active === null ? [] : neighbors[active],
        visited: visited.slice(), labels: labels.slice(), core: core.slice(),
      });
    }

    snapshot(null, "Ready", "Choose an unvisited point and inspect its neighborhood.");
    for (let i = 0; i < points.length; i += 1) {
      if (visited[i]) continue;
      visited[i] = true;
      core[i] = neighbors[i].length >= minSamples;
      if (!core[i]) {
        snapshot(i, "Noise for now", `Point ${i + 1} is not core and is not yet in a cluster. Mark it as noise for now.`);
        continue;
      }

      const cluster = clusterCount;
      clusterCount += 1;
      labels[i] = cluster;
      const queue = [];
      const queued = new Set();
      function enqueue(indices) {
        indices.forEach((index) => {
          if (index !== i && !queued.has(index)) {
            queued.add(index);
            queue.push(index);
          }
        });
      }
      enqueue(neighbors[i]);
      snapshot(i, "Core", `Point ${i + 1} starts cluster ${cluster + 1}. Add its neighbors to the expansion queue.`, queue.length);

      while (queue.length) {
        const j = queue.shift();
        if (visited[j]) {
          if (labels[j] === -1) {
            labels[j] = cluster;
            snapshot(j, "Border", `Point ${j + 1} was marked noise earlier. It is reachable from a core point, so it becomes a border point of cluster ${cluster + 1}.`, queue.length);
          }
          continue;
        }
        visited[j] = true;
        if (labels[j] === -1) labels[j] = cluster;
        core[j] = neighbors[j].length >= minSamples;
        if (core[j]) {
          enqueue(neighbors[j]);
          snapshot(j, "Core", `Point ${j + 1} is core. Its neighbors can extend cluster ${cluster + 1}.`, queue.length);
        } else {
          snapshot(j, "Border", `Point ${j + 1} is not core, but it is reachable from a core point. It joins cluster ${cluster + 1} without expanding it.`, queue.length);
        }
      }
    }
    snapshot(null, "Complete", "All points have been visited. Any points still marked noise are the final noise points.");
    return { trace, neighbors };
  }

  if (typeof module !== "undefined" && module.exports) module.exports = { buildTrace };
  if (typeof document === "undefined") return;
  const root = document.getElementById("dbscan-explorer");
  if (!root) return;

  const find = (id) => root.querySelector(`#${id}`);
  const elements = {
    epsilon: find("dbscan-epsilon"), min: find("dbscan-min-samples"),
    epsilonValue: find("dbscan-epsilon-value"), minValue: find("dbscan-min-value"),
    previous: find("dbscan-previous"), next: find("dbscan-next"), play: find("dbscan-play"), reset: find("dbscan-reset"),
    progress: find("dbscan-progress"), plot: find("dbscan-plot"), action: find("dbscan-action"),
    neighborCount: find("dbscan-neighbors"), kind: find("dbscan-kind"), clusterCount: find("dbscan-clusters"),
    noiseCount: find("dbscan-noise-count"), explanation: find("dbscan-explanation"),
  };
  if (Object.values(elements).some((value) => !value)) return;

  const NS = "http://www.w3.org/2000/svg";
  const COLORS = ["#347fc0", "#e47c35", "#8b62ae", "#21977c", "#bc5570", "#b59b2b", "#4e7d92", "#915b45"];
  const state = { points: [], trace: [], step: 0, timer: null };
  // A square plotting area preserves the circular geometry of the data.
  const view = { left: 140, right: 560, top: 42, bottom: 462, xmin: -1.25, xmax: 1.25, ymin: -1.25, ymax: 1.25 };
  const scale = (view.right - view.left) / (view.xmax - view.xmin);
  const px = (x) => view.left + (x - view.xmin) * scale;
  const py = (y) => view.bottom - (y - view.ymin) * scale;

  function svg(name, attributes = {}) {
    const node = document.createElementNS(NS, name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  }
  function text(target, value, x, y, attributes = {}) {
    const node = svg("text", { x, y, ...attributes });
    node.textContent = value;
    target.append(node);
  }
  function stop() {
    if (state.timer !== null) window.clearInterval(state.timer);
    state.timer = null;
  }
  function drawPlot(event) {
    const plot = elements.plot;
    plot.replaceChildren();
    const title = svg("title");
    title.textContent = "DBSCAN clustering of the fixed concentric-circle dataset";
    plot.append(title);
    [-1, 0, 1].forEach((tick) => {
      plot.append(svg("line", { x1: px(tick), x2: px(tick), y1: view.top, y2: view.bottom, class: "dbscan-grid" }));
      text(plot, String(tick), px(tick), 487, { class: "dbscan-tick", "text-anchor": "middle" });
    });
    [-1, 0, 1].forEach((tick) => {
      plot.append(svg("line", { x1: view.left, x2: view.right, y1: py(tick), y2: py(tick), class: "dbscan-grid" }));
      text(plot, String(tick), 128, py(tick) + 4, { class: "dbscan-tick", "text-anchor": "end" });
    });
    plot.append(svg("line", { x1: view.left, x2: view.right, y1: view.bottom, y2: view.bottom, class: "dbscan-axis" }));
    plot.append(svg("line", { x1: view.left, x2: view.left, y1: view.top, y2: view.bottom, class: "dbscan-axis" }));
    text(plot, "x₁", (view.left + view.right) / 2, 510, { class: "dbscan-axis-label", "text-anchor": "middle" });
    text(plot, "x₂", 91, (view.top + view.bottom) / 2, { class: "dbscan-axis-label", "text-anchor": "middle", transform: `rotate(-90 91 ${(view.top + view.bottom) / 2})` });

    if (event.active !== null) {
      const point = state.points[event.active];
      plot.append(svg("circle", { cx: px(point.x), cy: py(point.y), r: Number(elements.epsilon.value) * scale, class: "dbscan-radius" }));
    }
    const neighborSet = new Set(event.neighbors);
    state.points.forEach((point, index) => {
      const assigned = event.labels[index] >= 0;
      const noise = event.visited[index] && !assigned;
      const color = assigned ? COLORS[event.labels[index] % COLORS.length] : noise ? "#d38953" : "#adbcc3";
      plot.append(svg("circle", { cx: px(point.x), cy: py(point.y), r: 4.5, fill: color, class: "dbscan-point" }));
      if (neighborSet.has(index)) {
        plot.append(svg("circle", { cx: px(point.x), cy: py(point.y), r: 7, class: "dbscan-neighbor-ring" }));
      }
    });
    if (event.active !== null) {
      const point = state.points[event.active];
      plot.append(svg("circle", { cx: px(point.x), cy: py(point.y), r: 10, class: "dbscan-active-ring" }));
    }
  }
  function render() {
    const event = state.trace[state.step];
    const finished = state.step === state.trace.length - 1;
    const noiseCount = event.labels.reduce((count, label, index) => count + (label === -1 && event.visited[index] ? 1 : 0), 0);
    elements.epsilonValue.textContent = Number(elements.epsilon.value).toFixed(2);
    elements.minValue.textContent = elements.min.value;
    elements.progress.textContent = `Step ${state.step} of ${state.trace.length - 1}`;
    elements.action.textContent = event.message;
    elements.neighborCount.textContent = event.active === null ? "—" : String(event.neighbors.length);
    elements.kind.textContent = event.kind;
    elements.clusterCount.textContent = String(event.clusterCount);
    elements.noiseCount.textContent = String(noiseCount);
    elements.explanation.textContent = finished
      ? `Finished: ${event.clusterCount} cluster${event.clusterCount === 1 ? "" : "s"} and ${noiseCount} final noise point${noiseCount === 1 ? "" : "s"}.`
      : event.active === null
        ? "Start by choosing an unvisited point and counting the points within ε."
        : `${event.neighbors.length} point${event.neighbors.length === 1 ? "" : "s"} within ε; ${event.queueSize} point${event.queueSize === 1 ? "" : "s"} waiting in the expansion queue.`;
    elements.previous.disabled = state.step === 0 || state.timer !== null;
    elements.next.disabled = finished || state.timer !== null;
    elements.play.disabled = finished && state.timer === null;
    elements.play.textContent = state.timer === null ? "Play" : "Pause";
    drawPlot(event);
  }
  function reset() {
    stop();
    state.trace = buildTrace(state.points, Number(elements.epsilon.value), Number(elements.min.value)).trace;
    state.step = 0;
    render();
  }
  elements.previous.addEventListener("click", () => { state.step -= 1; render(); });
  elements.next.addEventListener("click", () => { state.step += 1; render(); });
  elements.reset.addEventListener("click", reset);
  elements.play.addEventListener("click", () => {
    if (state.timer !== null) { stop(); render(); return; }
    state.timer = window.setInterval(() => {
      if (state.step < state.trace.length - 1) state.step += 1;
      if (state.step === state.trace.length - 1) stop();
      render();
    }, 110);
    render();
  });
  elements.epsilon.addEventListener("input", reset);
  elements.min.addEventListener("input", reset);

  fetch("../data/dbscan-concentric-circles.csv")
    .then((response) => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.text(); })
    .then((data) => {
      state.points = data.trim().split(/\r?\n/).slice(1).map((line) => {
        const [x, y] = line.split(",").map(Number); return { x, y };
      });
      if (!state.points.length || state.points.some((point) => !Number.isFinite(point.x) || !Number.isFinite(point.y))) throw new Error("Invalid point data");
      reset();
    })
    .catch(() => {
      elements.action.textContent = "The concentric-circle dataset could not be loaded.";
      [elements.previous, elements.next, elements.play, elements.reset].forEach((button) => { button.disabled = true; });
    });
})();
