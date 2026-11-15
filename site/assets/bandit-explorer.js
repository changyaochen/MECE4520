(() => {
  const ARMS = [
    { name: "A", mean: 0.31, spread: 0.25, color: "#4e79a7" },
    { name: "B", mean: 0.54, spread: 0.27, color: "#f28e2b" },
    { name: "C", mean: 0.76, spread: 0.22, color: "#59a14f" },
    { name: "D", mean: 0.43, spread: 0.26, color: "#af7aa1" },
  ];
  const POLICY_COLORS = { greedy: "#8764a0", epsilon: "#347b9e", ucb: "#31946b" };
  const COMPARE_TRIALS = 200;
  const TRACE_TRIALS = 100;

  class SeededRandom {
    constructor(seed) { this.state = seed >>> 0; }
    uniform() {
      this.state = (Math.imul(1664525, this.state) + 1013904223) >>> 0;
      return this.state / 2 ** 32;
    }
    normal() {
      const u = Math.max(this.uniform(), Number.EPSILON);
      const v = this.uniform();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
  }

  function boundedReward(mean, spread, normal) {
    return Math.max(0, Math.min(1, mean + spread * normal));
  }

  function buildRewardStreams(length = COMPARE_TRIALS) {
    return ARMS.map((arm, index) => {
      const rng = new SeededRandom(104729 + index * 7919);
      return Array.from({ length }, () => boundedReward(arm.mean, arm.spread, rng.normal()));
    });
  }

  function chooseTied(candidates, rng) {
    return candidates[Math.floor(rng.uniform() * candidates.length)];
  }

  function maxCandidates(values) {
    const max = Math.max(...values);
    return values.reduce((indices, value, index) => {
      if (value === max || Math.abs(value - max) < 1e-12) indices.push(index);
      return indices;
    }, []);
  }

  function createSimulation(policy, epsilon, c, streamLength = COMPARE_TRIALS) {
    return {
      policy,
      epsilon,
      c,
      step: 0,
      estimates: ARMS.map(() => 0),
      counts: ARMS.map(() => 0),
      history: [],
      streams: buildRewardStreams(streamLength),
      actionRng: new SeededRandom(81571),
      totalReward: 0,
    };
  }

  function greedyAction(state) {
    return chooseTied(maxCandidates(state.estimates), state.actionRng);
  }

  function ucbScores(state) {
    const t = Math.max(1, state.step);
    return state.estimates.map((estimate, index) => {
      if (state.counts[index] === 0) return Infinity;
      return estimate + state.c * Math.sqrt(Math.log(t + 1) / state.counts[index]);
    });
  }

  function selectAction(state) {
    if (state.policy === "greedy") return { action: greedyAction(state), mode: "greedy" };
    if (state.policy === "epsilon") {
      if (state.actionRng.uniform() < state.epsilon) {
        return { action: Math.floor(state.actionRng.uniform() * ARMS.length), mode: "explore" };
      }
      return { action: greedyAction(state), mode: "exploit" };
    }
    const scores = ucbScores(state);
    return { action: chooseTied(maxCandidates(scores), state.actionRng), mode: "ucb", scores };
  }

  function stepSimulation(state) {
    const choice = selectAction(state);
    const previousCount = state.counts[choice.action];
    const reward = state.streams[choice.action][previousCount];
    state.counts[choice.action] += 1;
    state.estimates[choice.action] += (reward - state.estimates[choice.action]) / state.counts[choice.action];
    state.totalReward += reward;
    state.step += 1;
    state.history.push({
      trial: state.step,
      action: choice.action,
      reward,
      runningAverage: state.totalReward / state.step,
      mode: choice.mode,
      scores: choice.scores || null,
    });
    return state.history.at(-1);
  }

  function runSimulation(policy, epsilon, c, trials = COMPARE_TRIALS) {
    const state = createSimulation(policy, epsilon, c, trials);
    for (let index = 0; index < trials; index += 1) stepSimulation(state);
    return state;
  }

  // Permit a lightweight test outside the browser.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { ARMS, createSimulation, stepSimulation, runSimulation, ucbScores };
  }
  if (typeof document === "undefined") return;

  const root = document.getElementById("bandit-explorer");
  if (!root) return;
  const get = (id) => document.getElementById(`bandit-${id}`);
  const svgNamespace = "http://www.w3.org/2000/svg";
  const trace = get("trace");
  const comparison = get("comparison");
  const next = get("next");
  const run = get("run");
  const reset = get("reset");
  const progress = get("progress");
  const epsilonInput = get("epsilon");
  const cInput = get("c");
  const epsilonValue = get("epsilon-value");
  const cValue = get("c-value");
  const reveal = get("reveal");
  const action = get("action");
  const explanation = get("explanation");
  const lastReward = get("last-reward");
  const runningAverage = get("running-average");
  const mostPulled = get("most-pulled");
  const armTable = get("arm-table");
  const policyButtons = [...root.querySelectorAll("[data-policy]")];
  let policy = "greedy";
  let simulation = createSimulation(policy, Number(epsilonInput.value), Number(cInput.value));
  let timer = null;
  let runTarget = null;

  function svg(name, attributes = {}) {
    const element = document.createElementNS(svgNamespace, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
    return element;
  }

  function addText(parent, value, x, y, attributes = {}) {
    const element = svg("text", { x, y, ...attributes });
    element.textContent = value;
    parent.append(element);
    return element;
  }

  function linePath(history, x, y, accessor) {
    return history.map((entry, index) => `${index ? "L" : "M"}${x(entry.trial)},${y(accessor(entry))}`).join(" ");
  }

  function drawAxes(target, dimensions, xMax) {
    const { width, height, left, right, top, bottom } = dimensions;
    const innerWidth = width - left - right;
    const innerHeight = height - top - bottom;
    const x = (value) => left + (value / xMax) * innerWidth;
    const y = (value) => top + (1 - value) * innerHeight;
    [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
      const py = y(tick);
      target.append(svg("line", { x1: left, x2: width - right, y1: py, y2: py, class: "bandit-grid" }));
      addText(target, tick.toFixed(2), left - 10, py + 4, { class: "bandit-tick", "text-anchor": "end" });
    });
    const xTicks = xMax <= 100 ? [0, 25, 50, 75, 100] : [0, 50, 100, 150, 200];
    xTicks.filter((tick) => tick <= xMax).forEach((tick) => {
      const px = x(tick);
      target.append(svg("line", { x1: px, x2: px, y1: top, y2: height - bottom, class: "bandit-grid" }));
      addText(target, tick, px, height - bottom + 20, { class: "bandit-tick", "text-anchor": "middle" });
    });
    target.append(svg("line", { x1: left, x2: width - right, y1: height - bottom, y2: height - bottom, class: "bandit-axis" }));
    target.append(svg("line", { x1: left, x2: left, y1: top, y2: height - bottom, class: "bandit-axis" }));
    addText(target, "Decision", left + innerWidth / 2, height - 10, { class: "bandit-axis-label", "text-anchor": "middle" });
    addText(target, "Reward", 17, top + innerHeight / 2, { class: "bandit-axis-label", "text-anchor": "middle", transform: `rotate(-90 17 ${top + innerHeight / 2})` });
    return { x, y };
  }

  function drawTrace() {
    trace.replaceChildren();
    const title = svg("title");
    title.textContent = "Observed rewards and running average reward";
    trace.append(title);
    const dimensions = { width: 760, height: 410, left: 58, right: 24, top: 28, bottom: 48 };
    const { x, y } = drawAxes(trace, dimensions, TRACE_TRIALS);
    const bestMean = Math.max(...ARMS.map((arm) => arm.mean));
    trace.append(svg("line", { x1: x(0), x2: x(TRACE_TRIALS), y1: y(bestMean), y2: y(bestMean), class: "bandit-reference-line" }));
    if (reveal.checked) addText(trace, `best true mean = ${bestMean.toFixed(2)}`, x(TRACE_TRIALS) - 3, y(bestMean) - 7, { class: "bandit-reveal-label", "text-anchor": "end" });
    if (!simulation.history.length) {
      addText(trace, "Make a decision to observe a noisy reward.", 380, 197, { class: "bandit-empty-label", "text-anchor": "middle" });
      return;
    }
    simulation.history.forEach((entry) => {
      trace.append(svg("circle", { cx: x(entry.trial), cy: y(entry.reward), r: 4.5, fill: ARMS[entry.action].color, stroke: "#fff", "stroke-width": 1.2 }));
    });
    trace.append(svg("path", { d: linePath(simulation.history, x, y, (entry) => entry.runningAverage), class: "bandit-running-line" }));
  }

  function scoreText(index) {
    if (policy === "ucb") {
      if (simulation.counts[index] === 0) return "∞";
      return ucbScores(simulation)[index].toFixed(2);
    }
    return simulation.estimates[index].toFixed(2);
  }

  function drawArmTable() {
    armTable.replaceChildren();
    const header = document.createElement("div");
    header.className = "bandit-arm-row bandit-arm-header";
    ["", "Arm", "Estimate Q", "Pulls", policy === "ucb" ? "UCB score" : "Rule score"].forEach((label) => {
      const element = document.createElement("span");
      element.textContent = label;
      header.append(element);
    });
    armTable.append(header);
    ARMS.forEach((arm, index) => {
      const row = document.createElement("div");
      row.className = "bandit-arm-row";
      const swatch = document.createElement("i");
      swatch.className = "bandit-arm-swatch";
      swatch.style.backgroundColor = arm.color;
      const name = document.createElement("span");
      name.innerHTML = reveal.checked ? `<strong>${arm.name}</strong> <small>μ=${arm.mean.toFixed(2)}</small>` : `<strong>${arm.name}</strong>`;
      const estimate = document.createElement("strong");
      estimate.textContent = simulation.estimates[index].toFixed(2);
      const count = document.createElement("strong");
      count.textContent = simulation.counts[index];
      const score = document.createElement("strong");
      score.className = "bandit-ucb-score";
      score.textContent = scoreText(index);
      row.append(swatch, name, estimate, count, score);
      armTable.append(row);
    });
  }

  function updateDetails() {
    const last = simulation.history.at(-1);
    progress.textContent = `Decision ${simulation.step}`;
    lastReward.textContent = last ? last.reward.toFixed(2) : "—";
    runningAverage.textContent = last ? last.runningAverage.toFixed(2) : "—";
    const maxCount = Math.max(...simulation.counts);
    mostPulled.textContent = maxCount ? ARMS[simulation.counts.indexOf(maxCount)].name : "—";
    if (!last) {
      action.textContent = "Awaiting the first reward";
      explanation.textContent = policy === "greedy"
        ? "All estimates are tied at zero, so the first arm is chosen at random. After that, greedy selects the arm with the largest observed estimate."
        : policy === "epsilon"
          ? `With probability ε = ${simulation.epsilon.toFixed(2)}, the policy explores a random arm. Otherwise, it chooses the arm with the largest estimate.`
          : "UCB gives an infinite score to each untried arm, so it samples every arm before comparing estimates and uncertainty bonuses.";
      return;
    }
    const chosen = ARMS[last.action].name;
    if (policy === "greedy") {
      action.textContent = `Greedy selected arm ${chosen}`;
      explanation.textContent = `Arm ${chosen} had the largest current estimate. The policy does not deliberately revisit uncertain arms.`;
    } else if (policy === "epsilon") {
      action.textContent = last.mode === "explore" ? `ε-greedy explored arm ${chosen}` : `ε-greedy exploited arm ${chosen}`;
      explanation.textContent = last.mode === "explore"
        ? `This decision was one of the ε = ${simulation.epsilon.toFixed(2)} random explorations.`
        : `This decision chose the currently best estimated arm; ε-greedy still explores on future decisions.`;
    } else {
      action.textContent = `UCB selected arm ${chosen}`;
      explanation.textContent = simulation.counts[last.action] === 1
        ? "UCB ensures that every arm is tried at least once before it relies on estimates."
        : "UCB balances the estimated reward with an uncertainty bonus, favoring arms that remain less explored.";
    }
  }

  function drawComparison() {
    comparison.replaceChildren();
    const title = svg("title");
    title.textContent = "Running average reward under three bandit policies";
    comparison.append(title);
    const dimensions = { width: 1020, height: 330, left: 56, right: 26, top: 28, bottom: 45 };
    const { x, y } = drawAxes(comparison, dimensions, COMPARE_TRIALS);
    const epsilon = Number(epsilonInput.value);
    const c = Number(cInput.value);
    ["greedy", "epsilon", "ucb"].forEach((policyName) => {
      const runState = runSimulation(policyName, epsilon, c, COMPARE_TRIALS);
      comparison.append(svg("path", { d: linePath(runState.history, x, y, (entry) => entry.runningAverage), class: "bandit-comparison-line", stroke: POLICY_COLORS[policyName] }));
    });
    const bestMean = Math.max(...ARMS.map((arm) => arm.mean));
    comparison.append(svg("line", { x1: x(0), x2: x(COMPARE_TRIALS), y1: y(bestMean), y2: y(bestMean), class: "bandit-reference-line" }));
  }

  function render() {
    policyButtons.forEach((button) => {
      const selected = button.dataset.policy === policy;
      button.classList.toggle("bandit-policy-selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    epsilonValue.textContent = Number(epsilonInput.value).toFixed(2);
    cValue.textContent = Number(cInput.value).toFixed(2);
    drawTrace();
    drawArmTable();
    updateDetails();
    drawComparison();
  }

  function stopRunning() {
    if (timer !== null) window.clearInterval(timer);
    timer = null;
    runTarget = null;
    run.textContent = "Run 25";
  }

  function resetSimulation() {
    stopRunning();
    simulation = createSimulation(policy, Number(epsilonInput.value), Number(cInput.value));
    render();
  }

  function makeDecision() {
    if (simulation.step >= TRACE_TRIALS) { stopRunning(); return; }
    stepSimulation(simulation);
    if (simulation.step >= TRACE_TRIALS || (runTarget !== null && simulation.step >= runTarget)) stopRunning();
    render();
  }

  policyButtons.forEach((button) => button.addEventListener("click", () => {
    policy = button.dataset.policy;
    resetSimulation();
  }));
  epsilonInput.addEventListener("input", resetSimulation);
  cInput.addEventListener("input", resetSimulation);
  reveal.addEventListener("change", render);
  next.addEventListener("click", () => { stopRunning(); makeDecision(); });
  reset.addEventListener("click", resetSimulation);
  run.addEventListener("click", () => {
    if (timer !== null) { stopRunning(); render(); return; }
    if (simulation.step >= TRACE_TRIALS) resetSimulation();
    runTarget = Math.min(simulation.step + 25, TRACE_TRIALS);
    run.textContent = "Pause";
    timer = window.setInterval(makeDecision, 150);
  });
  render();
})();
