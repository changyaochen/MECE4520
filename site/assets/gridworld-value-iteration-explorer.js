(() => {
  const GRID_SIZE = 5;
  const GAMMA = 0.9;
  const TOLERANCE = 1e-4;
  const TIE_TOLERANCE = 1e-10;
  const ACTIONS = [
    { key: "up", row: -1, column: 0, label: "Up", arrow: "↑" },
    { key: "right", row: 0, column: 1, label: "Right", arrow: "→" },
    { key: "down", row: 1, column: 0, label: "Down", arrow: "↓" },
    { key: "left", row: 0, column: -1, label: "Left", arrow: "←" },
  ];
  const SPECIAL = [
    { label: "A", source: { row: 0, column: 1 }, targetLabel: "A′", target: { row: 4, column: 1 }, reward: 10, className: "special-a" },
    { label: "B", source: { row: 0, column: 3 }, targetLabel: "B′", target: { row: 2, column: 3 }, reward: 5, className: "special-b" },
  ];
  const sameState = (a, b) => a.row === b.row && a.column === b.column;
  const specialSource = (s) => SPECIAL.find((item) => sameState(s, item.source));
  const specialTarget = (s) => SPECIAL.find((item) => sameState(s, item.target));
  const zeroValues = () => Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0));

  function transition(state, actionKey) {
    const action = ACTIONS.find((candidate) => candidate.key === actionKey);
    if (!action) throw new Error(`Unknown action: ${actionKey}`);
    const special = specialSource(state);
    if (special) return { action, nextState: { ...special.target }, reward: special.reward };
    const next = { row: state.row + action.row, column: state.column + action.column };
    if (next.row < 0 || next.row >= GRID_SIZE || next.column < 0 || next.column >= GRID_SIZE) {
      return { action, nextState: { ...state }, reward: -1 };
    }
    return { action, nextState: next, reward: 0 };
  }

  function actionValues(state, values) {
    return ACTIONS.map((action) => {
      const outcome = transition(state, action.key);
      return { ...outcome, value: outcome.reward + GAMMA * values[outcome.nextState.row][outcome.nextState.column] };
    });
  }

  function greedyActions(state, values) {
    const outcomes = actionValues(state, values);
    const best = Math.max(...outcomes.map((item) => item.value));
    return outcomes.filter((item) => Math.abs(item.value - best) < TIE_TOLERANCE);
  }

  function bellmanSweep(values) {
    const nextValues = zeroValues();
    let maxChange = 0;
    for (let row = 0; row < GRID_SIZE; row += 1) {
      for (let column = 0; column < GRID_SIZE; column += 1) {
        const best = Math.max(...actionValues({ row, column }, values).map((item) => item.value));
        nextValues[row][column] = best;
        maxChange = Math.max(maxChange, Math.abs(best - values[row][column]));
      }
    }
    return { values: nextValues, maxChange };
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { ACTIONS, GAMMA, GRID_SIZE, TOLERANCE, SPECIAL, zeroValues, transition, actionValues, greedyActions, bellmanSweep };
  }
  if (typeof document === "undefined") return;
  const root = document.getElementById("gridworld-value-iteration-explorer");
  if (!root) return;
  const get = (id) => document.getElementById(`gridworld-value-${id}`);
  const valueGrid = get("value-grid");
  const policyGrid = get("policy-grid");
  const selectedLabel = get("selected");
  const actionsBody = get("actions");
  const result = get("result");
  const status = get("status");
  const reset = get("reset");
  const step = get("step");
  const run = get("run");
  let values = zeroValues();
  let selected = { row: 2, column: 2 };
  let sweep = 0;
  let maxChange = null;
  let timer = null;
  let converged = false;

  const coordinate = (state) => `(${state.row}, ${state.column})`;
  function stateLabel(state) {
    const source = specialSource(state);
    const target = specialTarget(state);
    return `${source ? `${source.label} ` : target ? `${target.targetLabel} ` : ""}${coordinate(state)}`;
  }
  function displayValue(value) { return (Math.abs(value) < 0.005 ? 0 : value).toFixed(2); }
  function appendSpan(parent, className, text) {
    const span = document.createElement("span");
    span.className = className;
    span.textContent = text;
    parent.append(span);
    return span;
  }
  function cellFor(state, type) {
    const source = specialSource(state);
    const target = specialTarget(state);
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = ["gridworld-iteration-cell", `gridworld-iteration-${type}-cell`, source?.className, target ? "target" : "", sameState(state, selected) ? "is-selected" : ""].filter(Boolean).join(" ");
    cell.setAttribute("role", "gridcell");
    cell.setAttribute("aria-selected", String(sameState(state, selected)));
    cell.addEventListener("click", () => { selected = { ...state }; render(); });
    appendSpan(cell, "gridworld-iteration-coordinate", coordinate(state));
    const marker = source?.label || target?.targetLabel;
    if (marker) appendSpan(cell, "gridworld-iteration-state-marker", marker);
    return cell;
  }
  function colorFor(value, maximum) {
    const dark = document.documentElement.classList.contains("quarto-dark") || document.body.classList.contains("quarto-dark");
    const base = dark ? [38, 54, 61] : [247, 251, 252];
    const end = value >= 0 ? (dark ? [33, 92, 64] : [165, 211, 174]) : (dark ? [47, 77, 98] : [161, 192, 207]);
    const ratio = Math.min(1, Math.abs(value) / maximum);
    return `rgb(${base.map((x, i) => Math.round(x + (end[i] - x) * ratio)).join(", ")})`;
  }
  function renderGrid(grid, type) {
    grid.replaceChildren();
    const maximum = Math.max(1, ...values.flat().map(Math.abs));
    for (let row = 0; row < GRID_SIZE; row += 1) {
      const gridRow = document.createElement("div");
      gridRow.className = "gridworld-iteration-row";
      gridRow.setAttribute("role", "row");
      for (let column = 0; column < GRID_SIZE; column += 1) {
        const state = { row, column };
        const cell = cellFor(state, type);
        if (type === "value") {
          cell.style.backgroundColor = colorFor(values[row][column], maximum);
          cell.setAttribute("aria-label", `${stateLabel(state)}; value ${displayValue(values[row][column])}`);
          appendSpan(cell, "gridworld-iteration-value", displayValue(values[row][column]));
        } else {
          const selectedActions = greedyActions(state, values);
          cell.setAttribute("aria-label", `${stateLabel(state)}; greedy actions ${selectedActions.map((item) => item.action.label).join(", ")}`);
          const layout = document.createElement("span");
          layout.className = `gridworld-iteration-action-layout${selectedActions.length === 1 ? " is-single" : ""}`;
          layout.setAttribute("aria-hidden", "true");
          selectedActions.forEach(({ action }) => appendSpan(layout, `gridworld-iteration-arrow gridworld-iteration-arrow-${action.key}`, action.arrow));
          cell.append(layout);
        }
        gridRow.append(cell);
      }
      grid.append(gridRow);
    }
  }
  function renderInspector() {
    selectedLabel.textContent = `Selected state: ${stateLabel(selected)}`;
    actionsBody.replaceChildren();
    const outcomes = actionValues(selected, values);
    const best = Math.max(...outcomes.map((outcome) => outcome.value));
    for (const outcome of outcomes) {
      const row = document.createElement("tr");
      if (Math.abs(outcome.value - best) < TIE_TOLERANCE) row.classList.add("gridworld-iteration-greedy");
      const cells = [`${outcome.action.arrow} ${outcome.action.label}`, stateLabel(outcome.nextState), outcome.reward > 0 ? `+${outcome.reward}` : String(outcome.reward), displayValue(values[outcome.nextState.row][outcome.nextState.column]), displayValue(outcome.value)];
      cells.forEach((text) => { const td = document.createElement("td"); td.textContent = text; row.append(td); });
      actionsBody.append(row);
    }
    result.textContent = `Next value at ${coordinate(selected)}: ${displayValue(best)} (using sweep ${sweep})`;
  }
  function render() {
    renderGrid(valueGrid, "value");
    renderGrid(policyGrid, "policy");
    renderInspector();
    status.textContent = `Sweep ${sweep} · max change ${maxChange === null ? "—" : maxChange.toExponential(2)}${converged ? " · converged" : ""}`;
    step.disabled = timer !== null || converged;
    run.disabled = timer === null && converged;
    run.textContent = timer === null ? "Run to convergence" : "Pause";
  }
  function stop() { if (timer !== null) window.clearInterval(timer); timer = null; }
  function oneSweep() {
    const update = bellmanSweep(values);
    values = update.values;
    maxChange = update.maxChange;
    sweep += 1;
    converged = maxChange < TOLERANCE;
    if (converged) stop();
    render();
  }
  reset.addEventListener("click", () => { stop(); values = zeroValues(); sweep = 0; maxChange = null; converged = false; render(); });
  step.addEventListener("click", oneSweep);
  run.addEventListener("click", () => {
    if (timer !== null) { stop(); render(); return; }
    oneSweep();
    if (!converged) timer = window.setInterval(oneSweep, 160);
    render();
  });
  window.addEventListener("pagehide", stop);
  new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  render();
})();
