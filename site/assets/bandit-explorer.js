(() => {
  // These values are generated reproducibly from the notebook's environment
  // seed (4_520), then held fixed while the agent learns.
  const DEFAULT_ARMS = [
    { name: "A", mean: 1, std: 1, color: "#4e79a7" },
    { name: "B", mean: 2, std: 1, color: "#f28e2b" },
    { name: "C", mean: 4, std: 1, color: "#59a14f" },
    { name: "D", mean: 3, std: 1, color: "#af7aa1" },
  ];
  // Start with the same reproducible four-arm environment used in the notebook.
  // Later "New simulation" clicks replace these values with a fresh bandit.
  const ARMS = DEFAULT_ARMS.map((arm) => ({ ...arm }));
  const POLICY_COLORS = { greedy: "#8764a0", epsilon: "#347b9e", ucb: "#31946b" };
  const COMPARE_TRIALS = 200;
  const AGGREGATE_RUNS = 300;
  // The notebook and this explorer both use a 200-decision horizon.
  const TRACE_TRIALS = COMPARE_TRIALS;
  let environmentIndex = 0;

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

  function regenerateArmDistributions() {
    environmentIndex += 1;
    const rng = new SeededRandom(225403 + environmentIndex * 104729);
    const bestArmIndex = Math.floor(rng.uniform() * ARMS.length);
    const means = ARMS.map(() => Number((1 + 2 * rng.uniform()).toFixed(1)));
    // Keep one clearly better arm, so each fresh example remains instructive.
    means[bestArmIndex] = Number((4 + rng.uniform()).toFixed(1));
    const nextArms = ARMS.map((arm, index) => ({
      ...arm,
      mean: means[index],
      std: 1,
    }));
    ARMS.splice(0, ARMS.length, ...nextArms);
  }

  function buildRewardStreams(length = COMPARE_TRIALS, replicate = 0) {
    return ARMS.map((arm, index) => {
      const rng = new SeededRandom(
        104729 + environmentIndex * 104729 + replicate * 196613 + index * 7919,
      );
      return Array.from({ length }, () => arm.mean + arm.std * rng.normal());
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

  function createSimulation(policy, epsilon, c, streamLength = COMPARE_TRIALS, replicate = 0) {
    return {
      policy,
      epsilon,
      c,
      step: 0,
      estimates: ARMS.map(() => 0),
      counts: ARMS.map(() => 0),
      history: [],
      streams: buildRewardStreams(streamLength, replicate),
      actionRng: new SeededRandom(81571 + environmentIndex * 5471 + replicate * 5471),
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

  function runSimulation(policy, epsilon, c, trials = COMPARE_TRIALS, replicate = 0) {
    const state = createSimulation(policy, epsilon, c, trials, replicate);
    for (let index = 0; index < trials; index += 1) stepSimulation(state);
    return state;
  }

  function trueBestArmIndex() {
    return ARMS.reduce(
      (bestIndex, arm, index) => arm.mean > ARMS[bestIndex].mean ? index : bestIndex,
      0,
    );
  }

  function averagePolicyBehavior(epsilon, c, runs = AGGREGATE_RUNS) {
    const bestArmIndex = trueBestArmIndex();
    return Object.fromEntries(["greedy", "epsilon", "ucb"].map((policy) => {
      const averageRewards = Array(COMPARE_TRIALS).fill(0);
      const bestArmRates = Array(COMPARE_TRIALS).fill(0);
      for (let replicate = 0; replicate < runs; replicate += 1) {
        const state = runSimulation(policy, epsilon, c, COMPARE_TRIALS, replicate);
        state.history.forEach((entry, index) => {
          averageRewards[index] += entry.runningAverage;
          bestArmRates[index] += Number(entry.action === bestArmIndex);
        });
      }
      return [policy, {
        averageReward: averageRewards.map((value) => value / runs),
        bestArmRate: bestArmRates.map((value) => value / runs),
      }];
    }));
  }

  // Permit a lightweight test outside the browser.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      ARMS,
      createSimulation,
      stepSimulation,
      runSimulation,
      ucbScores,
      regenerateArmDistributions,
      averagePolicyBehavior,
    };
  }
  if (typeof document === "undefined") return;

  const root = document.getElementById("bandit-explorer");
  if (!root) return;
  const get = (id) => document.getElementById(`bandit-${id}`);
  const svgNamespace = "http://www.w3.org/2000/svg";
  const trace = get("trace");
  const comparison = get("comparison");
  const bestArmRatePlot = get("best-arm-rate");
  const next = get("next");
  const run = get("run");
  const runAll = get("run-all");
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
  const answerTable = get("answer-table");
  const answerDetails = root.querySelector(".bandit-answer");
  const referenceLegend = get("reference-legend");
  const comparisonReferenceLegend = get("comparison-reference-legend");
  const policyButtons = [...root.querySelectorAll("[data-policy]")];
  let policy = "greedy";
  let simulation = createSimulation(policy, Number(epsilonInput.value), Number(cInput.value));
  let timer = null;
  let runTarget = null;
  let runningMode = null;
  let aggregateCache = { key: null, data: null };
  let renderedAggregateKey = null;

  function syncStickyOffset() {
    const header = document.getElementById("quarto-header");
    const headerHeight = header ? header.getBoundingClientRect().height : 0;
    root.style.setProperty("--bandit-navbar-height", `${headerHeight}px`);
  }

  const header = document.getElementById("quarto-header");
  if (header && "ResizeObserver" in window) {
    new ResizeObserver(syncStickyOffset).observe(header);
  }
  window.addEventListener("resize", syncStickyOffset);
  syncStickyOffset();

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

  function curvePath(values, x, y) {
    return values.map((value, index) => `${index ? "L" : "M"}${x(index + 1)},${y(value)}`).join(" ");
  }

  function drawAxes(target, dimensions, xMax, yMin, yMax, {
    yLabel = "Reward",
    formatY = (value) => Number(value.toFixed(2)).toString(),
  } = {}) {
    const { width, height, left, right, top, bottom } = dimensions;
    const innerWidth = width - left - right;
    const innerHeight = height - top - bottom;
    const x = (value) => left + (value / xMax) * innerWidth;
    const y = (value) => top + (yMax - value) / (yMax - yMin) * innerHeight;
    Array.from({ length: 5 }, (_, index) => yMin + index * (yMax - yMin) / 4).forEach((tick) => {
      const py = y(tick);
      target.append(svg("line", { x1: left, x2: width - right, y1: py, y2: py, class: "bandit-grid" }));
      addText(target, formatY(tick), left - 10, py + 4, { class: "bandit-tick", "text-anchor": "end" });
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
    addText(target, yLabel, 17, top + innerHeight / 2, { class: "bandit-axis-label", "text-anchor": "middle", transform: `rotate(-90 17 ${top + innerHeight / 2})` });
    return { x, y };
  }

  function drawTrace() {
    trace.replaceChildren();
    const title = svg("title");
    title.textContent = "Observed rewards and running average reward";
    trace.append(title);
    const dimensions = { width: 760, height: 410, left: 58, right: 24, top: 28, bottom: 48 };
    // The rewards are unbounded Normal draws. Show roughly 3.5 standard
    // deviations on either side of the current environment's means.
    const yMin = Math.min(...ARMS.map((arm) => arm.mean - 3.5 * arm.std)) - 0.25;
    const yMax = Math.max(...ARMS.map((arm) => arm.mean + 3.5 * arm.std)) + 0.25;
    const { x, y } = drawAxes(trace, dimensions, TRACE_TRIALS, yMin, yMax);
    const bestMean = Math.max(...ARMS.map((arm) => arm.mean));
    if (reveal.checked) {
      trace.append(svg("line", { x1: x(0), x2: x(TRACE_TRIALS), y1: y(bestMean), y2: y(bestMean), class: "bandit-reference-line" }));
      addText(trace, `best true mean = ${bestMean.toFixed(2)}`, x(TRACE_TRIALS) - 3, y(bestMean) - 7, { class: "bandit-reveal-label", "text-anchor": "end" });
    }
    if (!simulation.history.length) {
      addText(trace, "Make a decision to observe a sampled reward.", 380, 197, { class: "bandit-empty-label", "text-anchor": "middle" });
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

  function drawAnswerTable() {
    answerTable.replaceChildren();
    const table = document.createElement("table");
    table.innerHTML = "<thead><tr><th>Arm</th><th>Reward distribution</th><th>Mean reward</th></tr></thead>";
    const body = document.createElement("tbody");
    const bestArmIndex = trueBestArmIndex();
    ARMS.forEach((arm, index) => {
      const row = document.createElement("tr");
      if (index === bestArmIndex) row.className = "bandit-best-arm";
      const bestLabel = index === bestArmIndex ? '<span class="bandit-best-label">Highest mean</span>' : "";
      row.innerHTML = `<th scope="row"><i class="bandit-arm-swatch" style="background:${arm.color}"></i>${arm.name}${bestLabel}</th><td><code>Normal(${arm.mean.toFixed(2)}, ${arm.std.toFixed(2)}²)</code></td><td>${arm.mean.toFixed(2)}</td>`;
      body.append(row);
    });
    table.append(body);
    answerTable.append(table);
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

  function getAggregateComparison(epsilon, c) {
    const key = `${environmentIndex}|${epsilon}|${c}`;
    if (aggregateCache.key !== key) {
      aggregateCache = { key, data: averagePolicyBehavior(epsilon, c) };
    }
    return aggregateCache;
  }

  function drawBestArmRate(policyBehavior) {
    bestArmRatePlot.replaceChildren();
    const title = svg("title");
    title.textContent = "Percent of simulations choosing the true best arm";
    bestArmRatePlot.append(title);
    const dimensions = { width: 520, height: 330, left: 68, right: 22, top: 28, bottom: 45 };
    const { x, y } = drawAxes(bestArmRatePlot, dimensions, COMPARE_TRIALS, 0, 100, {
      yLabel: "Percent of simulations",
      formatY: (value) => `${Math.round(value)}%`,
    });
    ["greedy", "epsilon", "ucb"].forEach((policyName) => {
      const values = policyBehavior[policyName].bestArmRate.map((value) => 100 * value);
      bestArmRatePlot.append(svg("path", {
        d: curvePath(values, x, y),
        class: "bandit-comparison-line",
        stroke: POLICY_COLORS[policyName],
      }));
    });
  }

  function drawComparison() {
    const epsilon = Number(epsilonInput.value);
    const c = Number(cInput.value);
    const aggregate = getAggregateComparison(epsilon, c);
    const renderKey = `${aggregate.key}|${reveal.checked}`;
    if (renderedAggregateKey === renderKey) return;

    comparison.replaceChildren();
    const title = svg("title");
    title.textContent = "Average reward over independent bandit simulations";
    comparison.append(title);
    const dimensions = { width: 520, height: 330, left: 62, right: 22, top: 28, bottom: 45 };
    const maximumMean = Math.max(...ARMS.map((arm) => arm.mean));
    const { x, y } = drawAxes(comparison, dimensions, COMPARE_TRIALS, -1, Math.max(5, Math.ceil(maximumMean + 1)), {
      yLabel: "Average reward",
    });
    ["greedy", "epsilon", "ucb"].forEach((policyName) => {
      comparison.append(svg("path", {
        d: curvePath(aggregate.data[policyName].averageReward, x, y),
        class: "bandit-comparison-line",
        stroke: POLICY_COLORS[policyName],
      }));
    });
    if (reveal.checked) {
      const bestMean = Math.max(...ARMS.map((arm) => arm.mean));
      comparison.append(svg("line", { x1: x(0), x2: x(COMPARE_TRIALS), y1: y(bestMean), y2: y(bestMean), class: "bandit-reference-line" }));
    }
    drawBestArmRate(aggregate.data);
    renderedAggregateKey = renderKey;
  }

  function render() {
    updateRunButtons();
    policyButtons.forEach((button) => {
      const selected = button.dataset.policy === policy;
      button.classList.toggle("bandit-policy-selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    epsilonValue.textContent = Number(epsilonInput.value).toFixed(2);
    cValue.textContent = Number(cInput.value).toFixed(2);
    referenceLegend.hidden = !reveal.checked;
    comparisonReferenceLegend.hidden = !reveal.checked;
    drawTrace();
    drawArmTable();
    drawAnswerTable();
    updateDetails();
    drawComparison();
  }

  function updateRunButtons() {
    const isRunning = timer !== null;
    run.textContent = isRunning && runningMode === "batch" ? "Pause" : "Run 25";
    runAll.textContent = isRunning && runningMode === "all" ? "Pause" : "Run all";
    run.disabled = isRunning && runningMode !== "batch";
    runAll.disabled = isRunning && runningMode !== "all";
  }

  function stopRunning() {
    if (timer !== null) window.clearInterval(timer);
    timer = null;
    runTarget = null;
    runningMode = null;
    updateRunButtons();
  }

  function resetSimulation({ newEnvironment = false } = {}) {
    stopRunning();
    if (newEnvironment) {
      regenerateArmDistributions();
      reveal.checked = false;
      answerDetails.open = false;
    }
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
  reset.addEventListener("click", () => resetSimulation({ newEnvironment: true }));
  function startRunning(mode) {
    if (timer !== null) { stopRunning(); render(); return; }
    if (simulation.step >= TRACE_TRIALS) resetSimulation();
    runTarget = mode === "batch"
      ? Math.min(simulation.step + 25, TRACE_TRIALS)
      : TRACE_TRIALS;
    runningMode = mode;
    updateRunButtons();
    timer = window.setInterval(makeDecision, mode === "all" ? 40 : 150);
  }
  run.addEventListener("click", () => startRunning("batch"));
  runAll.addEventListener("click", () => startRunning("all"));
  render();
})();
