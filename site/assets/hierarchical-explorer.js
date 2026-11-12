(() => {
  const COLORS = ["#347fc0", "#e47c35", "#8b62ae", "#21977c", "#bc5570", "#b59b2b", "#4e7d92", "#915b45"];

  function distance(pointA, pointB) {
    return Math.hypot(pointA.x - pointB.x, pointA.y - pointB.y);
  }

  // For single linkage, cluster distance is the smallest point-to-point
  // distance across the two clusters. Keep the witness pair for the plot.
  function singleLinkageDistance(points, clusterA, clusterB) {
    let best = { distance: Infinity, leftPoint: null, rightPoint: null };
    clusterA.leaves.forEach((leftPoint) => {
      clusterB.leaves.forEach((rightPoint) => {
        const candidate = distance(points[leftPoint], points[rightPoint]);
        if (candidate < best.distance) best = { distance: candidate, leftPoint, rightPoint };
      });
    });
    return best;
  }

  function buildTrace(points) {
    const nodes = points.map((_, id) => ({ id, leaves: [id], height: 0, mergeIndex: 0 }));
    const nodeById = new Map(nodes.map((node) => [node.id, node]));
    let active = nodes.slice();
    const trace = [{ activeIds: active.map((node) => node.id), merge: null }];

    for (let mergeIndex = 1; active.length > 1; mergeIndex += 1) {
      let best = null;
      for (let i = 0; i < active.length; i += 1) {
        for (let j = i + 1; j < active.length; j += 1) {
          const candidate = singleLinkageDistance(points, active[i], active[j]);
          if (best === null || candidate.distance < best.distance) {
            best = { ...candidate, left: active[i], right: active[j] };
          }
        }
      }
      const [first, second] = best.left.leaves[0] < best.right.leaves[0]
        ? [best.left, best.right] : [best.right, best.left];
      const merged = {
        id: nodes.length,
        leaves: [...best.left.leaves, ...best.right.leaves].sort((a, b) => a - b),
        left: first.id,
        right: second.id,
        height: best.distance,
        mergeIndex,
      };
      nodes.push(merged);
      nodeById.set(merged.id, merged);
      active = active.filter((node) => node !== best.left && node !== best.right);
      active.push(merged);
      active.sort((nodeA, nodeB) => nodeA.leaves[0] - nodeB.leaves[0]);
      trace.push({
        activeIds: active.map((node) => node.id),
        merge: {
          left: best.left.id,
          right: best.right.id,
          node: merged.id,
          distance: best.distance,
          witness: [best.leftPoint, best.rightPoint],
        },
      });
    }
    return { nodes, nodeById, trace };
  }

  if (typeof module !== "undefined" && module.exports) module.exports = { buildTrace, singleLinkageDistance };
  if (typeof document === "undefined") return;

  const root = document.getElementById("hierarchical-explorer");
  if (!root) return;
  const find = (id) => root.querySelector(`#${id}`);
  const elements = {
    previous: find("hierarchical-previous"), next: find("hierarchical-next"), play: find("hierarchical-play"),
    reset: find("hierarchical-reset"), progress: find("hierarchical-progress"), plot: find("hierarchical-plot"),
    tree: find("hierarchical-dendrogram"), action: find("hierarchical-action"),
    explanation: find("hierarchical-explanation"), clusterCount: find("hierarchical-cluster-count"),
    distance: find("hierarchical-distance"), mergeCount: find("hierarchical-merge-count"),
  };
  if (Object.values(elements).some((element) => !element)) return;

  const NS = "http://www.w3.org/2000/svg";
  const state = { points: [], nodes: [], nodeById: new Map(), trace: [], step: 0, timer: null };
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

  function pointLayout(points) {
    const left = 66; const right = 26; const top = 26; const bottom = 58;
    const width = 620; const height = 480;
    const xMin = Math.min(...points.map((point) => point.x)) - 0.22;
    const xMax = Math.max(...points.map((point) => point.x)) + 0.22;
    const yMin = Math.min(...points.map((point) => point.y)) - 0.22;
    const yMax = Math.max(...points.map((point) => point.y)) + 0.22;
    const scale = Math.min((width - left - right) / (xMax - xMin), (height - top - bottom) / (yMax - yMin));
    const chartWidth = (xMax - xMin) * scale;
    const chartHeight = (yMax - yMin) * scale;
    const originX = left + ((width - left - right) - chartWidth) / 2;
    const originY = top + ((height - top - bottom) - chartHeight) / 2;
    return {
      x: (value) => originX + (value - xMin) * scale,
      y: (value) => originY + chartHeight - (value - yMin) * scale,
      left: originX, right: originX + chartWidth, top: originY, bottom: originY + chartHeight,
      xMin, xMax, yMin, yMax,
    };
  }

  function drawPoints(snapshot) {
    const plot = elements.plot;
    plot.replaceChildren();
    const title = svg("title");
    title.textContent = "Current single-linkage clusters in the two-moon dataset";
    plot.append(title);
    const layout = pointLayout(state.points);
    const xTicks = [-1, 0, 1, 2].filter((tick) => tick >= layout.xMin && tick <= layout.xMax);
    const yTicks = [-0.5, 0, 0.5, 1].filter((tick) => tick >= layout.yMin && tick <= layout.yMax);
    xTicks.forEach((tick) => {
      plot.append(svg("line", { x1: layout.x(tick), x2: layout.x(tick), y1: layout.top, y2: layout.bottom, class: "hierarchical-grid" }));
      text(plot, String(tick), layout.x(tick), layout.bottom + 22, { class: "hierarchical-tick", "text-anchor": "middle" });
    });
    yTicks.forEach((tick) => {
      plot.append(svg("line", { x1: layout.left, x2: layout.right, y1: layout.y(tick), y2: layout.y(tick), class: "hierarchical-grid" }));
      text(plot, String(tick), layout.left - 11, layout.y(tick) + 4, { class: "hierarchical-tick", "text-anchor": "end" });
    });
    plot.append(svg("line", { x1: layout.left, x2: layout.right, y1: layout.bottom, y2: layout.bottom, class: "hierarchical-axis" }));
    plot.append(svg("line", { x1: layout.left, x2: layout.left, y1: layout.top, y2: layout.bottom, class: "hierarchical-axis" }));
    text(plot, "x₁", (layout.left + layout.right) / 2, 466, { class: "hierarchical-axis-label", "text-anchor": "middle" });
    text(plot, "x₂", 21, (layout.top + layout.bottom) / 2, { class: "hierarchical-axis-label", "text-anchor": "middle", transform: `rotate(-90 21 ${(layout.top + layout.bottom) / 2})` });

    const merge = snapshot.merge;
    if (merge) {
      const [leftPoint, rightPoint] = merge.witness;
      plot.append(svg("line", {
        x1: layout.x(state.points[leftPoint].x), y1: layout.y(state.points[leftPoint].y),
        x2: layout.x(state.points[rightPoint].x), y2: layout.y(state.points[rightPoint].y), class: "hierarchical-pair-line",
      }));
    }
    const pointFills = state.points.map(() => "#9eafb7");
    snapshot.activeIds.forEach((clusterId, index) => {
      const node = state.nodeById.get(clusterId);
      const fill = snapshot.activeIds.length <= COLORS.length ? COLORS[index] : node.leaves.length > 1 ? COLORS[clusterId % COLORS.length] : "#9eafb7";
      node.leaves.forEach((pointId) => { pointFills[pointId] = fill; });
    });
    const leftSet = merge ? new Set(state.nodeById.get(merge.left).leaves) : new Set();
    const rightSet = merge ? new Set(state.nodeById.get(merge.right).leaves) : new Set();
    state.points.forEach((point, index) => {
      const className = leftSet.has(index) ? "hierarchical-pair-left" : rightSet.has(index) ? "hierarchical-pair-right" : "hierarchical-point";
      plot.append(svg("circle", { cx: layout.x(point.x), cy: layout.y(point.y), r: leftSet.has(index) || rightSet.has(index) ? 5.6 : 4.5, fill: pointFills[index], class: className }));
    });
  }

  function treeLayout() {
    const root = state.nodes[state.nodes.length - 1];
    function leavesInOrder(nodeId) {
      const node = state.nodeById.get(nodeId);
      if (!Object.hasOwn(node, "left")) return [node.id];
      return [...leavesInOrder(node.left), ...leavesInOrder(node.right)];
    }
    const leaves = leavesInOrder(root.id);
    const leafX = new Map();
    // Reserve enough left margin to keep the vertical title clear of tick labels.
    const width = 620; const left = 72; const right = 22; const top = 32; const bottom = 436;
    const tickX = left - 10; const axisLabelX = 18;
    leaves.forEach((leaf, index) => leafX.set(leaf, left + (width - right - left) * index / Math.max(1, leaves.length - 1)));
    const maxHeight = Math.max(root.height, 1e-9);
    const position = new Map();
    function locate(nodeId) {
      if (position.has(nodeId)) return position.get(nodeId);
      const node = state.nodeById.get(nodeId);
      const result = Object.hasOwn(node, "left")
        ? { x: (locate(node.left).x + locate(node.right).x) / 2, y: bottom - node.height / maxHeight * (bottom - top) }
        : { x: leafX.get(node.id), y: bottom };
      position.set(nodeId, result);
      return result;
    }
    state.nodes.forEach((node) => locate(node.id));
    return { position, top, bottom, maxHeight, left, right: width - right, tickX, axisLabelX };
  }

  function drawTree(snapshot) {
    const tree = elements.tree;
    tree.replaceChildren();
    const title = svg("title");
    title.textContent = "Dendrogram of the completed single-linkage hierarchy, revealed one merge at a time";
    tree.append(title);
    const layout = treeLayout();
    [0, layout.maxHeight / 2, layout.maxHeight].forEach((value) => {
      const y = layout.bottom - value / layout.maxHeight * (layout.bottom - layout.top);
      tree.append(svg("line", { x1: layout.left, x2: layout.right, y1: y, y2: y, class: "hierarchical-grid" }));
      text(tree, value.toFixed(value < 1 ? 2 : 1), layout.tickX, y + 4, { class: "hierarchical-tick", "text-anchor": "end" });
    });
    text(tree, "merge distance", layout.axisLabelX, (layout.top + layout.bottom) / 2, { class: "hierarchical-axis-label", "text-anchor": "middle", transform: `rotate(-90 ${layout.axisLabelX} ${(layout.top + layout.bottom) / 2})` });
    state.nodes.filter((node) => !Object.hasOwn(node, "left")).forEach((leaf) => {
      const point = layout.position.get(leaf.id);
      tree.append(svg("circle", { cx: point.x, cy: point.y, r: 2.3, class: "hierarchical-tree-leaf" }));
    });
    state.nodes.filter((node) => Object.hasOwn(node, "left") && node.mergeIndex <= state.step).forEach((node) => {
      const left = layout.position.get(node.left); const right = layout.position.get(node.right); const parent = layout.position.get(node.id);
      const current = snapshot.merge && snapshot.merge.node === node.id;
      tree.append(svg("path", { d: `M ${left.x} ${left.y} V ${parent.y} H ${right.x} V ${right.y}`, class: current ? "hierarchical-tree-branch hierarchical-tree-branch-current" : "hierarchical-tree-branch" }));
    });
    if (snapshot.merge) {
      const parent = layout.position.get(snapshot.merge.node);
      tree.append(svg("circle", { cx: parent.x, cy: parent.y, r: 4.2, class: "hierarchical-tree-leaf-current" }));
    }
  }

  function render() {
    const snapshot = state.trace[state.step];
    const totalMerges = state.trace.length - 1;
    const final = state.step === totalMerges;
    elements.progress.textContent = `Merge ${state.step} of ${totalMerges}`;
    elements.clusterCount.textContent = String(snapshot.activeIds.length);
    elements.distance.textContent = snapshot.merge ? snapshot.merge.distance.toFixed(3) : "—";
    elements.mergeCount.textContent = String(state.step);
    if (state.step === 0) {
      elements.action.textContent = "Start with one cluster per observation.";
      elements.explanation.textContent = "Single linkage will repeatedly find the two clusters with the closest pair of observations.";
    } else if (final) {
      elements.action.textContent = "The full hierarchy is complete.";
      elements.explanation.textContent = "To produce two final clusters, cut the dendrogram just below its final merge. That keeps the two curved moons separate.";
    } else {
      elements.action.textContent = `Merge ${state.step}: join the highlighted clusters.`;
      elements.explanation.textContent = `Their closest pair is ${snapshot.merge.distance.toFixed(3)} units apart. After the merge, ${snapshot.activeIds.length} clusters remain.`;
    }
    elements.previous.disabled = state.step === 0 || state.timer !== null;
    elements.next.disabled = final || state.timer !== null;
    elements.play.disabled = final && state.timer === null;
    elements.play.textContent = state.timer === null ? "Play" : "Pause";
    drawPoints(snapshot);
    drawTree(snapshot);
  }

  function reset() {
    stop();
    state.step = 0;
    render();
  }
  function forward() {
    if (state.step < state.trace.length - 1) state.step += 1;
    if (state.step === state.trace.length - 1) stop();
    render();
  }
  elements.previous.addEventListener("click", () => { stop(); state.step -= 1; render(); });
  elements.next.addEventListener("click", () => { stop(); forward(); });
  elements.reset.addEventListener("click", reset);
  elements.play.addEventListener("click", () => {
    if (state.timer !== null) { stop(); render(); return; }
    state.timer = window.setInterval(forward, 620);
    render();
  });
  [elements.previous, elements.next, elements.play, elements.reset].forEach((button) => { button.disabled = true; });

  fetch("../data/hierarchical-two-moons.csv")
    .then((response) => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.text(); })
    .then((csv) => {
      state.points = csv.trim().split(/\r?\n/).slice(1).map((line) => {
        const [x, y] = line.split(",").map(Number); return { x, y };
      });
      if (!state.points.length || state.points.some((point) => !Number.isFinite(point.x) || !Number.isFinite(point.y))) throw new Error("Invalid point data");
      const result = buildTrace(state.points);
      state.nodes = result.nodes; state.nodeById = result.nodeById; state.trace = result.trace;
      [elements.previous, elements.next, elements.play, elements.reset].forEach((button) => { button.disabled = false; });
      reset();
    })
    .catch(() => {
      elements.action.textContent = "The two-moon dataset could not be loaded.";
      elements.explanation.textContent = "Reload the page to try again.";
    });
})();
