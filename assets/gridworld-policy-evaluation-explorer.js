(() => {
  const GRID_SIZE = 5;
  const GAMMA = 0.9;
  const POLICY_PROBABILITY = 0.25;
  const CONVERGENCE_TOLERANCE = 1e-4;
  const ACTIONS = [
    { key: "up", row: -1, column: 0, label: "Up", arrow: "↑" },
    { key: "right", row: 0, column: 1, label: "Right", arrow: "→" },
    { key: "down", row: 1, column: 0, label: "Down", arrow: "↓" },
    { key: "left", row: 0, column: -1, label: "Left", arrow: "←" },
  ];
  const SPECIAL_TRANSITIONS = [
    {
      label: "A",
      source: { row: 0, column: 1 },
      targetLabel: "A′",
      target: { row: 4, column: 1 },
      reward: 10,
      className: "special-a",
    },
    {
      label: "B",
      source: { row: 0, column: 3 },
      targetLabel: "B′",
      target: { row: 2, column: 3 },
      reward: 5,
      className: "special-b",
    },
  ];

  const sameState = (first, second) => first.row === second.row && first.column === second.column;
  const copyState = (state) => ({ row: state.row, column: state.column });
  const inGrid = (state) => state.row >= 0 && state.row < GRID_SIZE
    && state.column >= 0 && state.column < GRID_SIZE;
  const sourceSpecialAt = (state) => SPECIAL_TRANSITIONS.find((special) => sameState(state, special.source));
  const targetSpecialAt = (state) => SPECIAL_TRANSITIONS.find((special) => sameState(state, special.target));

  function zeroValues() {
    return Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0));
  }

  function transition(state, actionKey) {
    const action = ACTIONS.find((candidate) => candidate.key === actionKey);
    if (!action) throw new Error(`Unknown Gridworld action: ${actionKey}`);

    const special = sourceSpecialAt(state);
    if (special) {
      return {
        action,
        nextState: copyState(special.target),
        reward: special.reward,
        kind: "special",
      };
    }

    const candidate = { row: state.row + action.row, column: state.column + action.column };
    if (!inGrid(candidate)) {
      return { action, nextState: copyState(state), reward: -1, kind: "boundary" };
    }
    return { action, nextState: candidate, reward: 0, kind: "ordinary" };
  }

  function backupForState(state, sourceValues) {
    const outcomes = ACTIONS.map((action) => {
      const outcome = transition(state, action.key);
      const continuation = outcome.reward + GAMMA * sourceValues[outcome.nextState.row][outcome.nextState.column];
      return { ...outcome, continuation, contribution: POLICY_PROBABILITY * continuation };
    });
    return {
      outcomes,
      value: outcomes.reduce((total, outcome) => total + outcome.contribution, 0),
    };
  }

  function bellmanSweep(sourceValues) {
    const nextValues = zeroValues();
    let maxChange = 0;
    for (let row = 0; row < GRID_SIZE; row += 1) {
      for (let column = 0; column < GRID_SIZE; column += 1) {
        const nextValue = backupForState({ row, column }, sourceValues).value;
        nextValues[row][column] = nextValue;
        maxChange = Math.max(maxChange, Math.abs(nextValue - sourceValues[row][column]));
      }
    }
    return { values: nextValues, maxChange };
  }

  // Keep the environment and Bellman updates testable without a browser.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      ACTIONS,
      GAMMA,
      GRID_SIZE,
      POLICY_PROBABILITY,
      SPECIAL_TRANSITIONS,
      backupForState,
      bellmanSweep,
      transition,
      zeroValues,
    };
  }
  if (typeof document === "undefined") return;

  const root = document.getElementById("gridworld-policy-evaluation-explorer");
  if (!root) return;
  const get = (id) => document.getElementById(`gridworld-evaluation-${id}`);
  const grid = get("grid");
  const reset = get("reset");
  const step = get("step");
  const run = get("run");
  const status = get("status");
  const selectedStateLabel = get("selected-state");
  const contributions = get("contributions");
  const resultLabel = get("result-label");
  const resultValue = get("result-value");
  const detailNote = get("detail-note");

  let values = zeroValues();
  let selectedState = { row: 2, column: 2 };
  let sweep = 0;
  let lastMaxChange = null;
  let timer = null;

  function coordinate(state) {
    return `(${state.row}, ${state.column})`;
  }

  function stateLabel(state) {
    const source = sourceSpecialAt(state);
    if (source) return `${source.label} ${coordinate(state)}`;
    const target = targetSpecialAt(state);
    if (target) return `${target.targetLabel} ${coordinate(state)}`;
    return coordinate(state);
  }

  function valueText(value) {
    const normalized = Math.abs(value) < 0.005 ? 0 : value;
    return normalized.toFixed(2);
  }

  function rewardText(reward) {
    return reward > 0 ? `+${reward}` : String(reward);
  }

  function mix(first, second, fraction) {
    return first.map((value, index) => Math.round(value + (second[index] - value) * fraction));
  }

  function valueColor(value, maxAbs) {
    const dark = document.documentElement.classList.contains("quarto-dark")
      || document.body.classList.contains("quarto-dark");
    const base = dark ? [38, 54, 61] : [247, 251, 252];
    const positive = dark ? [33, 92, 64] : [165, 211, 174];
    const negative = dark ? [47, 77, 98] : [161, 192, 207];
    const strength = Math.min(1, Math.abs(value) / maxAbs);
    return `rgb(${mix(base, value >= 0 ? positive : negative, strength).join(", ")})`;
  }

  function cellClass(state) {
    const names = ["gridworld-evaluation-cell"];
    const source = sourceSpecialAt(state);
    const target = targetSpecialAt(state);
    if (source) names.push(source.className);
    if (target) names.push("target");
    if (sameState(state, selectedState)) names.push("is-selected");
    return names.join(" ");
  }

  function appendText(parent, className, text) {
    const element = document.createElement("span");
    element.className = className;
    element.textContent = text;
    parent.append(element);
    return element;
  }

  function renderGrid() {
    grid.replaceChildren();
    const maxAbs = Math.max(1, ...values.flat().map((value) => Math.abs(value)));
    for (let row = 0; row < GRID_SIZE; row += 1) {
      const gridRow = document.createElement("div");
      gridRow.className = "gridworld-evaluation-row";
      gridRow.setAttribute("role", "row");
      for (let column = 0; column < GRID_SIZE; column += 1) {
        const state = { row, column };
        const cell = document.createElement("button");
        cell.type = "button";
        cell.className = cellClass(state);
        cell.setAttribute("role", "gridcell");
        cell.setAttribute("aria-selected", String(sameState(state, selectedState)));
        cell.setAttribute("aria-label", `${stateLabel(state)}; current value ${valueText(values[row][column])}`);
        cell.title = `Inspect the Bellman backup for ${stateLabel(state)}`;
        cell.style.backgroundColor = valueColor(values[row][column], maxAbs);

        appendText(cell, "gridworld-evaluation-coordinate", coordinate(state));
        appendText(cell, "gridworld-evaluation-value", valueText(values[row][column]));
        const source = sourceSpecialAt(state);
        const target = targetSpecialAt(state);
        if (source) appendText(cell, "gridworld-evaluation-special", source.label);
        if (target) appendText(cell, "gridworld-evaluation-special", target.targetLabel);
        cell.addEventListener("click", () => {
          selectedState = copyState(state);
          render();
        });
        gridRow.append(cell);
      }
      grid.append(gridRow);
    }
  }

  function appendCell(row, text, className = "") {
    const cell = document.createElement("td");
    cell.textContent = text;
    if (className) cell.className = className;
    row.append(cell);
  }

  function renderDetails() {
    const backup = backupForState(selectedState, values);
    selectedStateLabel.textContent = `Selected state: ${stateLabel(selectedState)}`;
    contributions.replaceChildren();
    backup.outcomes.forEach((outcome) => {
      const row = document.createElement("tr");
      appendCell(row, `${outcome.action.arrow} ${outcome.action.label}`, "gridworld-evaluation-action");
      appendCell(row, stateLabel(outcome.nextState));
      appendCell(
        row,
        rewardText(outcome.reward),
        outcome.reward > 0 ? "gridworld-evaluation-reward-positive"
          : outcome.reward < 0 ? "gridworld-evaluation-reward-negative" : "",
      );
      appendCell(row, valueText(outcome.contribution));
      contributions.append(row);
    });
    resultLabel.replaceChildren();
    const valueSymbol = document.createElement("i");
    valueSymbol.textContent = "V";
    const valueIndex = document.createElement("sub");
    valueIndex.textContent = sweep + 1;
    resultLabel.append(valueSymbol, valueIndex, document.createTextNode(coordinate(selectedState)));
    resultValue.textContent = valueText(backup.value);
    detailNote.innerHTML = `Each contribution is 0.25 × [<i>r</i> + 0.9 <i>V</i><sub>${sweep}</sub>(<i>s</i>′)]. All four use the same current grid <i>V</i><sub>${sweep}</sub>.`;
  }

  function updateControls() {
    const isRunning = timer !== null;
    step.disabled = isRunning;
    run.textContent = isRunning ? "Pause" : "Run to convergence";
    run.disabled = !isRunning && lastMaxChange !== null && lastMaxChange < CONVERGENCE_TOLERANCE;
    const deltaText = lastMaxChange === null ? "—" : lastMaxChange.toExponential(2);
    status.textContent = `Sweep ${sweep} · max change ${deltaText}`;
  }

  function render() {
    renderGrid();
    renderDetails();
    updateControls();
  }

  function stopRunning() {
    if (timer !== null) window.clearInterval(timer);
    timer = null;
  }

  function takeSweep() {
    const result = bellmanSweep(values);
    values = result.values;
    lastMaxChange = result.maxChange;
    sweep += 1;
    if (lastMaxChange < CONVERGENCE_TOLERANCE) stopRunning();
    render();
  }

  function resetEvaluation() {
    stopRunning();
    values = zeroValues();
    sweep = 0;
    lastMaxChange = null;
    render();
  }

  reset.addEventListener("click", resetEvaluation);
  step.addEventListener("click", takeSweep);
  run.addEventListener("click", () => {
    if (timer !== null) {
      stopRunning();
      render();
      return;
    }
    timer = window.setInterval(takeSweep, 180);
    render();
  });

  new MutationObserver(render).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class"],
  });
  render();
})();
