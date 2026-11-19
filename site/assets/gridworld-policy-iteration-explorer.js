(() => {
  const GRID_SIZE = 5;
  const GAMMA = 0.9;
  const EVALUATION_TOLERANCE = 1e-5;
  const POLICY_TIE_TOLERANCE = 1e-10;
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
  const stateKey = (state) => `${state.row},${state.column}`;
  const inGrid = (state) => state.row >= 0 && state.row < GRID_SIZE
    && state.column >= 0 && state.column < GRID_SIZE;
  const sourceSpecialAt = (state) => SPECIAL_TRANSITIONS.find((special) => sameState(state, special.source));
  const targetSpecialAt = (state) => SPECIAL_TRANSITIONS.find((special) => sameState(state, special.target));

  function zeroValues() {
    return Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0));
  }

  function uniformPolicy() {
    return Array.from(
      { length: GRID_SIZE },
      () => Array.from({ length: GRID_SIZE }, () => Array(ACTIONS.length).fill(1 / ACTIONS.length)),
    );
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

  function actionValues(state, values) {
    return ACTIONS.map((action) => {
      const outcome = transition(state, action.key);
      const oneStepValue = values === null
        ? null
        : outcome.reward + GAMMA * values[outcome.nextState.row][outcome.nextState.column];
      return { ...outcome, oneStepValue };
    });
  }

  function evaluatePolicy(policy, tolerance = EVALUATION_TOLERANCE) {
    let values = zeroValues();
    let maxChange = Infinity;
    let sweeps = 0;

    while (maxChange >= tolerance && sweeps < 10_000) {
      const nextValues = zeroValues();
      maxChange = 0;
      for (let row = 0; row < GRID_SIZE; row += 1) {
        for (let column = 0; column < GRID_SIZE; column += 1) {
          const state = { row, column };
          const expectedValue = actionValues(state, values).reduce(
            (total, outcome, actionIndex) => total + policy[row][column][actionIndex] * outcome.oneStepValue,
            0,
          );
          nextValues[row][column] = expectedValue;
          maxChange = Math.max(maxChange, Math.abs(expectedValue - values[row][column]));
        }
      }
      values = nextValues;
      sweeps += 1;
    }

    if (sweeps === 10_000) throw new Error("Policy evaluation did not converge.");
    return { values, sweeps, maxChange };
  }

  function greedyImprove(policy, values) {
    const nextPolicy = uniformPolicy();
    const changedStates = [];

    for (let row = 0; row < GRID_SIZE; row += 1) {
      for (let column = 0; column < GRID_SIZE; column += 1) {
        const state = { row, column };
        const outcomes = actionValues(state, values);
        const bestValue = Math.max(...outcomes.map((outcome) => outcome.oneStepValue));
        const bestActionIndices = outcomes
          .map((outcome, actionIndex) => ({ outcome, actionIndex }))
          .filter(({ outcome }) => Math.abs(outcome.oneStepValue - bestValue) < POLICY_TIE_TOLERANCE)
          .map(({ actionIndex }) => actionIndex);
        const probability = 1 / bestActionIndices.length;
        nextPolicy[row][column] = ACTIONS.map((_, actionIndex) => (
          bestActionIndices.includes(actionIndex) ? probability : 0
        ));

        const changed = nextPolicy[row][column].some(
          (value, actionIndex) => Math.abs(value - policy[row][column][actionIndex]) > POLICY_TIE_TOLERANCE,
        );
        if (changed) changedStates.push(state);
      }
    }

    return { policy: nextPolicy, changedStates, stable: changedStates.length === 0 };
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      ACTIONS,
      EVALUATION_TOLERANCE,
      GAMMA,
      GRID_SIZE,
      SPECIAL_TRANSITIONS,
      actionValues,
      evaluatePolicy,
      greedyImprove,
      transition,
      uniformPolicy,
      zeroValues,
    };
  }
  if (typeof document === "undefined") return;

  const root = document.getElementById("gridworld-policy-iteration-explorer");
  if (!root) return;
  const get = (id) => document.getElementById(`gridworld-iteration-${id}`);
  const policyGrid = get("policy-grid");
  const valueGrid = get("value-grid");
  const policyCaption = get("policy-caption");
  const valueCaption = get("value-caption");
  const selectedStateLabel = get("selected-state");
  const inspectorNote = get("inspector-note");
  const actionValueRows = get("action-values");
  const status = get("status");
  const reset = get("reset");
  const evaluate = get("evaluate");
  const improve = get("improve");
  const run = get("run");

  let policy = uniformPolicy();
  let policyIndex = 0;
  let values = null;
  let valuesForPolicyIndex = null;
  let evaluationSummary = null;
  let selectedState = { row: 2, column: 2 };
  let recentChanges = new Set();
  let phase = "needs-evaluation";
  let statusMessage = "Policy π₀ is ready to evaluate.";
  let timer = null;

  function coordinate(state) {
    return `(${state.row}, ${state.column})`;
  }

  function policyLabel(index) {
    const subscripts = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"];
    return `π${String(index).split("").map((digit) => subscripts[Number(digit)]).join("")}`;
  }

  function stateLabel(state) {
    const source = sourceSpecialAt(state);
    if (source) return `${source.label} ${coordinate(state)}`;
    const target = targetSpecialAt(state);
    if (target) return `${target.targetLabel} ${coordinate(state)}`;
    return coordinate(state);
  }

  function stateMarker(state) {
    const source = sourceSpecialAt(state);
    if (source) return source.label;
    const target = targetSpecialAt(state);
    return target ? target.targetLabel : "";
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

  function appendText(parent, className, text) {
    const element = document.createElement("span");
    element.className = className;
    element.textContent = text;
    parent.append(element);
    return element;
  }

  function buildCell(state, type) {
    const source = sourceSpecialAt(state);
    const target = targetSpecialAt(state);
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = [
      "gridworld-iteration-cell",
      type === "policy" ? "gridworld-iteration-policy-cell" : "gridworld-iteration-value-cell",
      source ? source.className : "",
      target ? "target" : "",
      sameState(state, selectedState) ? "is-selected" : "",
      type === "policy" && recentChanges.has(stateKey(state)) ? "changed" : "",
      type === "value" && values !== null && valuesForPolicyIndex !== policyIndex ? "is-stale" : "",
    ].filter(Boolean).join(" ");
    cell.setAttribute("role", "gridcell");
    cell.setAttribute("aria-selected", String(sameState(state, selectedState)));
    cell.addEventListener("click", () => {
      selectedState = copyState(state);
      render();
    });
    return cell;
  }

  function renderPolicyGrid() {
    policyGrid.replaceChildren();
    for (let row = 0; row < GRID_SIZE; row += 1) {
      const gridRow = document.createElement("div");
      gridRow.className = "gridworld-iteration-row";
      gridRow.setAttribute("role", "row");
      for (let column = 0; column < GRID_SIZE; column += 1) {
        const state = { row, column };
        const cell = buildCell(state, "policy");
        const selectedActions = ACTIONS.filter((_, actionIndex) => policy[row][column][actionIndex] > POLICY_TIE_TOLERANCE);
        const actionNames = selectedActions.map((action) => action.label).join(", ");
        cell.setAttribute("aria-label", `${stateLabel(state)}; policy actions ${actionNames}`);
        cell.title = `Inspect policy at ${stateLabel(state)}`;
        appendText(cell, "gridworld-iteration-coordinate", coordinate(state));
        const actionLayout = document.createElement("span");
        actionLayout.className = [
          "gridworld-iteration-action-layout",
          selectedActions.length === 1 ? "is-single" : "",
        ].filter(Boolean).join(" ");
        actionLayout.setAttribute("aria-hidden", "true");
        selectedActions.forEach((action) => {
          appendText(
            actionLayout,
            `gridworld-iteration-arrow gridworld-iteration-arrow-${action.key}`,
            action.arrow,
          );
        });
        cell.append(actionLayout);
        const marker = stateMarker(state);
        if (marker) appendText(cell, "gridworld-iteration-state-marker", marker);
        gridRow.append(cell);
      }
      policyGrid.append(gridRow);
    }
  }

  function renderValueGrid() {
    valueGrid.replaceChildren();
    valueGrid.classList.toggle("is-stale", values !== null && valuesForPolicyIndex !== policyIndex);
    const maxAbs = values === null ? 1 : Math.max(1, ...values.flat().map((value) => Math.abs(value)));
    for (let row = 0; row < GRID_SIZE; row += 1) {
      const gridRow = document.createElement("div");
      gridRow.className = "gridworld-iteration-row";
      gridRow.setAttribute("role", "row");
      for (let column = 0; column < GRID_SIZE; column += 1) {
        const state = { row, column };
        const cell = buildCell(state, "value");
        const value = values === null ? null : values[row][column];
        cell.style.backgroundColor = value === null ? "" : valueColor(value, maxAbs);
        cell.setAttribute("aria-label", value === null
          ? `${stateLabel(state)}; value has not been evaluated`
          : `${stateLabel(state)}; value ${valueText(value)}`);
        cell.title = `Inspect ${stateLabel(state)}`;
        appendText(cell, "gridworld-iteration-coordinate", coordinate(state));
        appendText(cell, `gridworld-iteration-value${value === null ? " is-empty" : ""}`, value === null ? "—" : valueText(value));
        const marker = stateMarker(state);
        if (marker) appendText(cell, "gridworld-iteration-state-marker", marker);
        gridRow.append(cell);
      }
      valueGrid.append(gridRow);
    }
  }

  function appendCell(row, text, className = "") {
    const cell = document.createElement("td");
    cell.textContent = text;
    if (className) cell.className = className;
    row.append(cell);
  }

  function renderInspector() {
    selectedStateLabel.textContent = `Selected state: ${stateLabel(selectedState)}`;
    actionValueRows.replaceChildren();
    const outcomes = actionValues(selectedState, values);
    const bestValue = values === null ? null : Math.max(...outcomes.map((outcome) => outcome.oneStepValue));

    outcomes.forEach((outcome, actionIndex) => {
      const row = document.createElement("tr");
      const isGreedy = bestValue !== null
        && Math.abs(outcome.oneStepValue - bestValue) < POLICY_TIE_TOLERANCE;
      const isChosen = policy[selectedState.row][selectedState.column][actionIndex] > POLICY_TIE_TOLERANCE;
      if (isGreedy) row.classList.add("gridworld-iteration-greedy");
      if (isChosen) row.classList.add("gridworld-iteration-policy-choice");
      appendCell(row, `${outcome.action.arrow} ${outcome.action.label}`, "gridworld-iteration-action");
      appendCell(row, stateLabel(outcome.nextState));
      appendCell(
        row,
        rewardText(outcome.reward),
        outcome.reward > 0 ? "gridworld-iteration-reward-positive"
          : outcome.reward < 0 ? "gridworld-iteration-reward-negative" : "",
      );
      appendCell(row, outcome.oneStepValue === null ? "—" : valueText(outcome.oneStepValue));
      appendCell(row, policy[selectedState.row][selectedState.column][actionIndex].toFixed(2));
      actionValueRows.append(row);
    });

    if (values === null) {
      inspectorNote.textContent = "Evaluate the current policy to compute r + 0.9 V(s′) for each action.";
    } else if (valuesForPolicyIndex === policyIndex) {
      inspectorNote.textContent = `Green rows maximize r + 0.9 V(s′) using the value function for ${policyLabel(policyIndex)}. The blue marker identifies actions selected by ${policyLabel(policyIndex)}.`;
    } else {
      inspectorNote.textContent = `The table uses the value function for ${policyLabel(valuesForPolicyIndex)}, which produced the new policy ${policyLabel(policyIndex)}. Evaluate ${policyLabel(policyIndex)} next.`;
    }
  }

  function renderCaptions() {
    if (policyIndex === 0 && phase === "needs-evaluation") {
      policyCaption.textContent = `${policyLabel(0)}: each arrow has probability 0.25.`;
    } else if (phase === "optimal") {
      policyCaption.textContent = `${policyLabel(policyIndex)} is stable: no greedy improvement changes an action.`;
    } else {
      policyCaption.textContent = `Current policy: ${policyLabel(policyIndex)}. Highlighted cells changed during the last improvement.`;
    }

    if (values === null) {
      valueCaption.textContent = `Evaluate ${policyLabel(policyIndex)} to fill this grid.`;
    } else if (valuesForPolicyIndex === policyIndex) {
      valueCaption.textContent = `Value function for ${policyLabel(policyIndex)}, evaluated in ${evaluationSummary.sweeps} Bellman sweeps.`;
    } else {
      valueCaption.textContent = `Shown: value function for ${policyLabel(valuesForPolicyIndex)}. These values are stale for ${policyLabel(policyIndex)}.`;
    }
  }

  function renderControls() {
    const isRunning = timer !== null;
    evaluate.disabled = isRunning || valuesForPolicyIndex === policyIndex || phase === "optimal";
    improve.disabled = isRunning || valuesForPolicyIndex !== policyIndex || phase === "optimal";
    run.textContent = isRunning ? "Pause" : "Run to optimum";
    run.disabled = !isRunning && phase === "optimal";
    status.textContent = statusMessage;
  }

  function render() {
    renderPolicyGrid();
    renderValueGrid();
    renderInspector();
    renderCaptions();
    renderControls();
  }

  function evaluateCurrentPolicy() {
    const result = evaluatePolicy(policy);
    values = result.values;
    valuesForPolicyIndex = policyIndex;
    evaluationSummary = result;
    recentChanges = new Set();
    phase = "ready-to-improve";
    statusMessage = `${policyLabel(policyIndex)} evaluated in ${result.sweeps} Bellman sweeps · max change ${result.maxChange.toExponential(2)}.`;
  }

  function improveCurrentPolicy() {
    const result = greedyImprove(policy, values);
    recentChanges = new Set(result.changedStates.map(stateKey));
    if (result.stable) {
      phase = "optimal";
      statusMessage = `${policyLabel(policyIndex)} is stable: no actions changed. It is the optimal policy.`;
      return;
    }
    policy = result.policy;
    policyIndex += 1;
    phase = "needs-evaluation";
    statusMessage = `${result.changedStates.length} states changed. ${policyLabel(policyIndex)} is ready to evaluate.`;
  }

  function stopRunning() {
    if (timer !== null) window.clearInterval(timer);
    timer = null;
  }

  function automaticStep() {
    if (valuesForPolicyIndex !== policyIndex) {
      evaluateCurrentPolicy();
    } else {
      improveCurrentPolicy();
    }
    if (phase === "optimal") stopRunning();
    render();
  }

  function resetExplorer() {
    stopRunning();
    policy = uniformPolicy();
    policyIndex = 0;
    values = null;
    valuesForPolicyIndex = null;
    evaluationSummary = null;
    recentChanges = new Set();
    phase = "needs-evaluation";
    statusMessage = "Policy π₀ is ready to evaluate.";
    render();
  }

  reset.addEventListener("click", resetExplorer);
  evaluate.addEventListener("click", () => {
    evaluateCurrentPolicy();
    render();
  });
  improve.addEventListener("click", () => {
    improveCurrentPolicy();
    render();
  });
  run.addEventListener("click", () => {
    if (timer !== null) {
      stopRunning();
      statusMessage = "Automatic run paused.";
      render();
      return;
    }
    automaticStep();
    if (phase !== "optimal") timer = window.setInterval(automaticStep, 900);
    render();
  });

  window.addEventListener("pagehide", stopRunning);
  new MutationObserver(render).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class"],
  });
  render();
})();
