(() => {
  const GRID_SIZE = 5;
  const INITIAL_STATE = { row: 2, column: 2 };
  const ACTIONS = {
    up: { row: -1, column: 0, label: "Up", arrow: "↑" },
    right: { row: 0, column: 1, label: "Right", arrow: "→" },
    down: { row: 1, column: 0, label: "Down", arrow: "↓" },
    left: { row: 0, column: -1, label: "Left", arrow: "←" },
  };
  const SPECIAL_TRANSITIONS = [
    {
      label: "A",
      source: { row: 0, column: 1 },
      targetLabel: "A′",
      target: { row: 4, column: 1 },
      reward: 10,
      color: "a",
    },
    {
      label: "B",
      source: { row: 0, column: 3 },
      targetLabel: "B′",
      target: { row: 2, column: 3 },
      reward: 5,
      color: "b",
    },
  ];

  const sameState = (first, second) => first.row === second.row && first.column === second.column;
  const copyState = (state) => ({ row: state.row, column: state.column });
  const sourceSpecialAt = (state) => SPECIAL_TRANSITIONS.find((special) => sameState(state, special.source));
  const targetSpecialAt = (state) => SPECIAL_TRANSITIONS.find((special) => sameState(state, special.target));
  const inGrid = (state) => state.row >= 0 && state.row < GRID_SIZE
    && state.column >= 0 && state.column < GRID_SIZE;

  function stateName(state) {
    const sourceSpecial = sourceSpecialAt(state);
    if (sourceSpecial) return sourceSpecial.label;
    const targetSpecial = targetSpecialAt(state);
    if (targetSpecial) return targetSpecial.targetLabel;
    return `row ${state.row + 1}, column ${state.column + 1}`;
  }

  function stateDescription(state) {
    const name = stateName(state);
    if (name.startsWith("row")) return name;
    return `${name} (row ${state.row + 1}, column ${state.column + 1})`;
  }

  function gridCoordinate(state) {
    return `(${state.row + 1}, ${state.column + 1})`;
  }

  function formatReward(reward) {
    return reward > 0 ? `+${reward}` : String(reward);
  }

  function transition(state, actionName) {
    const action = ACTIONS[actionName];
    if (!action) throw new Error(`Unknown Gridworld action: ${actionName}`);

    const special = sourceSpecialAt(state);
    if (special) {
      return {
        state: copyState(state),
        action: actionName,
        nextState: copyState(special.target),
        reward: special.reward,
        kind: "special",
        special,
      };
    }

    const candidate = {
      row: state.row + action.row,
      column: state.column + action.column,
    };
    if (!inGrid(candidate)) {
      return {
        state: copyState(state),
        action: actionName,
        nextState: copyState(state),
        reward: -1,
        kind: "boundary",
      };
    }
    return {
      state: copyState(state),
      action: actionName,
      nextState: candidate,
      reward: 0,
      kind: "ordinary",
    };
  }

  // Keep the environment dynamics independently testable from the browser UI.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { ACTIONS, GRID_SIZE, INITIAL_STATE, SPECIAL_TRANSITIONS, transition };
  }
  if (typeof document === "undefined") return;

  const root = document.getElementById("gridworld-environment-explorer");
  if (!root) return;
  const get = (id) => document.getElementById(`gridworld-${id}`);
  const grid = get("grid");
  const status = get("status");
  const stateValue = get("state-value");
  const actionValue = get("action-value");
  const nextStateValue = get("next-state-value");
  const rewardValue = get("reward-value");
  const rule = get("transition-rule");
  const actionButtons = [...root.querySelectorAll("[data-action]")];
  let currentState = copyState(INITIAL_STATE);
  let lastTransition = null;

  function cellLabel(state) {
    const sourceSpecial = sourceSpecialAt(state);
    if (sourceSpecial) return `${sourceSpecial.label}, special state: every action goes to ${sourceSpecial.targetLabel} with reward ${formatReward(sourceSpecial.reward)}`;
    const targetSpecial = targetSpecialAt(state);
    if (targetSpecial) return `${targetSpecial.targetLabel}, destination of special state ${targetSpecial.label}`;
    return `row ${state.row + 1}, column ${state.column + 1}`;
  }

  function addCellText(cell, className, value) {
    const element = document.createElement("span");
    element.className = className;
    element.textContent = value;
    cell.append(element);
  }

  function svgElement(name, attributes = {}) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
    return element;
  }

  function gridCenter(state) {
    const cellSize = 500 / GRID_SIZE;
    return {
      x: (state.column + 0.5) * cellSize,
      y: (state.row + 0.5) * cellSize,
    };
  }

  function addTeleportOverlay() {
    const overlay = svgElement("svg", {
      class: "gridworld-teleport-overlay",
      viewBox: "0 0 500 500",
      "aria-hidden": "true",
      focusable: "false",
    });
    const definitions = svgElement("defs");
    overlay.append(definitions);

    SPECIAL_TRANSITIONS.forEach((special) => {
      const markerId = `gridworld-teleport-arrow-${special.color}`;
      const marker = svgElement("marker", {
        id: markerId,
        viewBox: "0 0 8 8",
        refX: 6.8,
        refY: 4,
        markerWidth: 7,
        markerHeight: 7,
        orient: "auto",
      });
      marker.append(svgElement("path", {
        d: "M 0 0 L 8 4 L 0 8 z",
        class: `gridworld-teleport-arrowhead gridworld-teleport-${special.color}`,
      }));
      definitions.append(marker);

      const source = gridCenter(special.source);
      const target = gridCenter(special.target);
      // Draw both arrows on the right side of their source/destination cells,
      // mirroring the conventional Gridworld diagram.  The longer A arrow
      // bends farther so it stays clear of its labels.
      const isA = special.color === "a";
      const startX = source.x + (isA ? 30 : 25);
      const endX = target.x + (isA ? 30 : 25);
      const startY = source.y + (isA ? 30 : 27);
      const endY = target.y - (isA ? 30 : 27);
      const controlX = source.x + (isA ? 150 : 95);
      const middleY = (startY + endY) / 2;
      const rewardX = source.x + (isA ? 60 : 42);
      const arc = svgElement("path", {
        d: `M ${startX} ${startY} C ${controlX} ${startY + 42}, ${controlX} ${endY - 42}, ${endX} ${endY}`,
        class: `gridworld-teleport-arc gridworld-teleport-${special.color}`,
        "marker-end": `url(#${markerId})`,
      });
      overlay.append(arc);

      const reward = svgElement("text", {
        x: rewardX,
        y: middleY,
        class: `gridworld-teleport-reward gridworld-teleport-${special.color}`,
        "text-anchor": "middle",
        "dominant-baseline": "middle",
      });
      reward.textContent = formatReward(special.reward);
      overlay.append(reward);
    });

    grid.append(overlay);
  }

  function renderGrid() {
    grid.replaceChildren();
    for (let row = 0; row < GRID_SIZE; row += 1) {
      const gridRow = document.createElement("div");
      gridRow.className = "gridworld-row";
      gridRow.setAttribute("role", "row");
      for (let column = 0; column < GRID_SIZE; column += 1) {
        const state = { row, column };
        const sourceSpecial = sourceSpecialAt(state);
        const targetSpecial = targetSpecialAt(state);
        const isCurrent = sameState(state, currentState);
        const cell = document.createElement("button");
        cell.type = "button";
        cell.className = [
          "gridworld-cell",
          sourceSpecial ? `gridworld-special-${sourceSpecial.color}` : "",
          targetSpecial ? "gridworld-target" : "",
          isCurrent ? "gridworld-current" : "",
        ].filter(Boolean).join(" ");
        cell.setAttribute("role", "gridcell");
        cell.setAttribute("aria-selected", String(isCurrent));
        cell.setAttribute("aria-label", `${cellLabel(state)}${isCurrent ? ", current agent state" : ""}`);
        cell.title = `Place the agent at ${cellLabel(state)}`;

        addCellText(cell, "gridworld-cell-coordinate", `(${row + 1}, ${column + 1})`);
        if (sourceSpecial) {
          addCellText(cell, "gridworld-cell-reward gridworld-cell-zero-reward", "0");
          addCellText(cell, "gridworld-cell-state-label", sourceSpecial.label);
        } else if (targetSpecial) {
          addCellText(cell, "gridworld-cell-reward gridworld-cell-zero-reward", "0");
          addCellText(cell, "gridworld-cell-state-label", targetSpecial.targetLabel);
        }
        else {
          addCellText(cell, "gridworld-cell-reward gridworld-cell-zero-reward", "0");
        }

        if (isCurrent) addCellText(cell, "gridworld-agent-marker", "🤖");
        cell.addEventListener("click", () => {
          currentState = copyState(state);
          lastTransition = null;
          render();
          status.textContent = `Agent placed at ${stateDescription(currentState)}. Choose an action.`;
        });
        gridRow.append(cell);
      }
      grid.append(gridRow);
    }
    addTeleportOverlay();
  }

  function renderTransition() {
    if (!lastTransition) {
      stateValue.textContent = gridCoordinate(currentState);
      actionValue.textContent = "—";
      nextStateValue.textContent = "—";
      rewardValue.textContent = "—";
      rule.textContent = "For an ordinary state, moving inside the grid gives reward 0.";
      return;
    }

    const action = ACTIONS[lastTransition.action];
    const { state, nextState, reward, kind, special } = lastTransition;
    stateValue.textContent = gridCoordinate(state);
    actionValue.textContent = `${action.arrow} ${action.label}`;
    nextStateValue.textContent = gridCoordinate(nextState);
    rewardValue.textContent = formatReward(reward);
    rewardValue.className = reward > 0 ? "gridworld-positive-reward"
      : reward < 0 ? "gridworld-negative-reward" : "";

    if (kind === "special") {
      rule.textContent = `${special.label} overrides the chosen direction: every action sends the agent to ${special.targetLabel} and yields ${formatReward(special.reward)}.`;
    } else if (kind === "boundary") {
      rule.textContent = "The selected direction would leave the grid, so the agent stays in the same state and receives −1.";
    } else {
      rule.textContent = "This is an ordinary in-grid move: the agent changes state and receives reward 0.";
    }
  }

  function render() {
    renderGrid();
    renderTransition();
  }

  function takeAction(actionName) {
    lastTransition = transition(currentState, actionName);
    currentState = copyState(lastTransition.nextState);
    render();
    status.textContent = `After ${ACTIONS[actionName].label.toLowerCase()}, the agent is at ${stateDescription(currentState)} with reward ${formatReward(lastTransition.reward)}.`;
  }

  get("reset").addEventListener("click", () => {
    currentState = copyState(INITIAL_STATE);
    lastTransition = null;
    render();
    status.textContent = "Agent reset to row 3, column 3.";
  });
  get("go-a").addEventListener("click", () => {
    currentState = copyState(SPECIAL_TRANSITIONS[0].source);
    lastTransition = null;
    render();
    status.textContent = "Agent placed at A. Choose any action.";
  });
  get("go-b").addEventListener("click", () => {
    currentState = copyState(SPECIAL_TRANSITIONS[1].source);
    lastTransition = null;
    render();
    status.textContent = "Agent placed at B. Choose any action.";
  });
  actionButtons.forEach((button) => {
    button.addEventListener("click", () => takeAction(button.dataset.action));
  });

  render();
})();
