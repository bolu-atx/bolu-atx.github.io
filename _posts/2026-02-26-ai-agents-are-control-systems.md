---
layout: post
title: "Your AI Agent Is a Control System (It Just Doesn't Know It Yet)"
date: 2026-02-26 10:00:00 -0700
tags: ai llm control-theory machine-learning
author: bolu-atx
categories: machine-learning
---

Back in grad school, I built a linear state-space MPC controller for my control theory class -- the classic quadratic-cost-over-a-finite-horizon setup. You model the plant dynamics as $x_{k+1} = Ax_k + Bu_k$, define a cost function that penalizes deviations from your target state and excessive control effort, then solve a quadratic program at each timestep to get the optimal sequence of future inputs. But here's the key: you only apply the *first* input. Then you measure the new state, shift the horizon forward, and solve the whole thing again from scratch.

The professor called this "receding horizon control." At the time I thought it was a neat trick for keeping chemical reactors from exploding. A decade later, I'm watching Claude Code do *exactly the same thing* -- observe state, predict outcomes, pick the best next action, re-evaluate -- except the "plant" is a codebase and the "control inputs" are bash commands.

This isn't a metaphor. It's the same math.

<!--more-->

## The Mapping: MPC Concepts in Agent Architectures

Let's make the translation explicit. Every component of a Model Predictive Controller has a direct counterpart in an AI coding agent:

| MPC Concept | Agent Equivalent | What It Means |
|:---|:---|:---|
| **Plant** | Execution environment | The bash shell, Python REPL, or IDE where actions get executed |
| **State vector** $x(t)$ | Context window | Everything the agent currently "knows" -- files read, errors seen, plan state |
| **Internal model** $f$ | LLM world model | The model's learned ability to predict what happens when you run a command |
| **Control input** $u(t)$ | Tool call / generated token | The specific action the agent takes to change the environment |
| **Prediction horizon** $N$ | Planning depth | How many steps ahead the agent reasons before committing |
| **Stage cost** $\ell$ | Process reward model | How good or bad each intermediate step looks |
| **Terminal cost** $V_f$ | Outcome reward model | Did the final result actually satisfy the user's goal? |
| **Moving Horizon Estimation** | Context folding / compaction | Compressing old history into a summary so the window doesn't overflow |

The core MPC optimization problem is:

$$
\min_{u_0, \ldots, u_{N-1}} \sum_{k=0}^{N-1} \ell(x_k, u_k) + V_f(x_N)
$$

$$
\text{subject to} \quad x_{k+1} = f(x_k, u_k)
$$

In plain English: find the sequence of actions that minimizes total cost (bad intermediate states + bad final outcome), subject to the constraint that each state follows from the previous one through the system dynamics.

For a coding agent, this translates to: *find the sequence of tool calls that minimizes wasted effort and errors, subject to the constraint that each state of the codebase follows logically from the previous edit*. The LLM's "world model" -- its understanding of how code, compilers, and file systems behave -- serves as $f$. The reward signal (did the tests pass? did the user accept the change?) serves as $\ell$ and $V_f$.

## The Agentic Loop = Receding Horizon Control

Let's look at how real agents implement this.

### Claude Code's Recursive Loop

Claude Code runs a tight five-step cycle that maps directly onto the MPC control loop:

1. **Normalize** -- compact and summarize the conversation history (= Moving Horizon Estimation)
2. **Infer** -- the LLM generates its predicted response given the current state (= forward prediction)
3. **Detect tool use** -- pause if the model wants to invoke a tool (= compute optimal control action)
4. **Execute** -- run the tool and collect output (= apply first control action)
5. **Recurse** -- feed results back, re-enter the loop (= shift horizon, re-solve)

<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script src="/assets/js/control-charts.js"></script>

<div id="receding-horizon" style="text-align:center; margin: 2em 0;"></div>

<script>
(function() {
  'use strict';

  var animTimer = null;

  function render() {
    if (animTimer) { clearInterval(animTimer); animTimer = null; }

    var container = d3.select('#receding-horizon');
    container.selectAll('*').remove();
    var c = ControlCharts.getColors();

    var width = 600, height = 360;
    var margin = { top: 30, right: 30, bottom: 46, left: 56 };
    var innerW = width - margin.left - margin.right;
    var innerH = height - margin.top - margin.bottom;

    var svg = container.append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .attr('width', '100%')
      .style('max-width', width + 'px')
      .style('display', 'block')
      .style('margin', '0 auto');

    svg.append('rect')
      .attr('width', width).attr('height', height)
      .attr('fill', c.bg).attr('rx', 8);

    var g = svg.append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    // Data: actual trajectory with disturbances
    var totalSteps = 20;
    var horizonLen = 5;
    var actual = [1.0];
    var rng = d3.randomNormal(0, 0.04);
    for (var i = 1; i <= totalSteps; i++) {
      var prev = actual[i - 1];
      var decay = prev * 0.92 + rng();
      // Disturbance events at steps 6 and 14
      if (i === 6) decay = prev + 0.18;
      if (i === 14) decay = prev + 0.14;
      actual.push(Math.max(0.02, Math.min(1.2, decay)));
    }

    var x = d3.scaleLinear().domain([0, totalSteps]).range([0, innerW]);
    var y = d3.scaleLinear().domain([0, 1.25]).range([innerH, 0]);

    // Grid
    y.ticks(5).forEach(function(v) {
      g.append('line')
        .attr('x1', 0).attr('x2', innerW)
        .attr('y1', y(v)).attr('y2', y(v))
        .attr('stroke', c.grid).attr('stroke-width', 0.5);
    });

    // Axes
    g.append('g')
      .attr('transform', 'translate(0,' + innerH + ')')
      .call(d3.axisBottom(x).ticks(10).tickSize(-4))
      .call(function(g) { g.select('.domain').attr('stroke', c.muted); })
      .call(function(g) { g.selectAll('.tick line').attr('stroke', c.muted); })
      .call(function(g) { g.selectAll('.tick text').attr('fill', c.text).attr('font-size', '11px'); });

    g.append('g')
      .call(d3.axisLeft(y).ticks(5).tickSize(-4))
      .call(function(g) { g.select('.domain').attr('stroke', c.muted); })
      .call(function(g) { g.selectAll('.tick line').attr('stroke', c.muted); })
      .call(function(g) { g.selectAll('.tick text').attr('fill', c.text).attr('font-size', '11px'); });

    // Axis labels
    svg.append('text')
      .attr('x', margin.left + innerW / 2).attr('y', height - 6)
      .attr('text-anchor', 'middle').attr('fill', c.text).attr('font-size', '12px')
      .text('Agent step');
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(margin.top + innerH / 2)).attr('y', 14)
      .attr('text-anchor', 'middle').attr('fill', c.text).attr('font-size', '12px')
      .text('Semantic distance to goal');

    // Elements that update per step
    var horizonRect = g.append('rect')
      .attr('fill', c.horizon).attr('opacity', 0.15)
      .attr('rx', 3);

    var actualPath = g.append('path')
      .attr('fill', 'none').attr('stroke', c.actual)
      .attr('stroke-width', 2.5).attr('stroke-linecap', 'round');

    var plannedPath = g.append('path')
      .attr('fill', 'none').attr('stroke', c.planned)
      .attr('stroke-width', 2).attr('stroke-dasharray', '6,4')
      .attr('stroke-linecap', 'round');

    var disturbanceGroup = g.append('g');

    var stepDot = g.append('circle')
      .attr('r', 5).attr('fill', c.actual).attr('stroke', c.bg).attr('stroke-width', 2);

    var stepLabel = g.append('text')
      .attr('fill', c.text).attr('font-size', '11px').attr('text-anchor', 'middle');

    var line = d3.line()
      .x(function(d, i) { return x(d[0]); })
      .y(function(d, i) { return y(d[1]); })
      .curve(d3.curveMonotoneX);

    // Legend
    var legendData = [
      { color: c.actual, dash: '', label: 'Actual trajectory' },
      { color: c.planned, dash: '6,4', label: 'Planned (horizon)' },
      { color: c.horizon, dash: '', label: 'Prediction window', rect: true },
      { color: c.disturbance, dash: '', label: 'Disturbance', marker: true },
    ];
    var legendX = margin.left + 8;
    var legendY = 14;
    legendData.forEach(function(item, i) {
      var lx = legendX + i * 138;
      if (item.rect) {
        svg.append('rect').attr('x', lx).attr('y', legendY - 6)
          .attr('width', 18).attr('height', 12).attr('rx', 2)
          .attr('fill', item.color).attr('opacity', 0.3);
      } else if (item.marker) {
        svg.append('text').attr('x', lx + 5).attr('y', legendY + 4)
          .attr('fill', item.color).attr('font-size', '14px').attr('font-weight', 'bold').text('\u26a1');
      } else {
        svg.append('line').attr('x1', lx).attr('x2', lx + 18)
          .attr('y1', legendY).attr('y2', legendY)
          .attr('stroke', item.color).attr('stroke-width', 2.5)
          .attr('stroke-dasharray', item.dash);
      }
      svg.append('text').attr('x', lx + 24).attr('y', legendY + 4)
        .attr('fill', c.text).attr('font-size', '11px').text(item.label);
    });

    // Replay button
    var btnG = svg.append('g')
      .attr('transform', 'translate(' + (width - 70) + ',' + (height - 24) + ')')
      .style('cursor', 'pointer')
      .on('click', function() { render(); });
    btnG.append('rect')
      .attr('width', 56).attr('height', 22).attr('rx', 4)
      .attr('fill', 'none').attr('stroke', c.muted).attr('stroke-width', 1);
    btnG.append('text')
      .attr('x', 28).attr('y', 15).attr('text-anchor', 'middle')
      .attr('fill', c.muted).attr('font-size', '11px').text('Replay');

    // Animate step by step
    var currentStep = 0;

    function drawStep() {
      if (currentStep > totalSteps) {
        clearInterval(animTimer);
        animTimer = null;
        return;
      }

      // Actual path up to current step
      var actualData = [];
      for (var i = 0; i <= currentStep; i++) {
        actualData.push([i, actual[i]]);
      }
      actualPath.attr('d', line(actualData));

      // Current dot
      stepDot.attr('cx', x(currentStep)).attr('cy', y(actual[currentStep]));
      stepLabel.attr('x', x(currentStep)).attr('y', y(actual[currentStep]) - 12)
        .text('step ' + currentStep);

      // Horizon window
      var hStart = currentStep;
      var hEnd = Math.min(currentStep + horizonLen, totalSteps);
      horizonRect
        .attr('x', x(hStart)).attr('y', 0)
        .attr('width', x(hEnd) - x(hStart))
        .attr('height', innerH);

      // Planned trajectory within horizon (smooth decay from current)
      if (currentStep < totalSteps) {
        var plannedData = [];
        var pVal = actual[currentStep];
        for (var j = currentStep; j <= hEnd; j++) {
          plannedData.push([j, pVal]);
          pVal = pVal * 0.88;
        }
        plannedPath.attr('d', line(plannedData));
      }

      // Disturbance markers
      disturbanceGroup.selectAll('*').remove();
      [6, 14].forEach(function(dStep) {
        if (dStep <= currentStep) {
          disturbanceGroup.append('text')
            .attr('x', x(dStep)).attr('y', y(actual[dStep]) - 14)
            .attr('text-anchor', 'middle')
            .attr('fill', c.disturbance).attr('font-size', '16px').attr('font-weight', 'bold')
            .text('\u26a1');
          disturbanceGroup.append('text')
            .attr('x', x(dStep)).attr('y', y(actual[dStep]) - 28)
            .attr('text-anchor', 'middle')
            .attr('fill', c.disturbance).attr('font-size', '9px')
            .text(dStep === 6 ? 'bug!' : 'dep conflict!');
        }
      });

      currentStep++;
    }

    drawStep();
    animTimer = setInterval(drawStep, 500);
  }

  render();
  document.documentElement.addEventListener('themechange', function() {
    render();
  });
})();
</script>

```mermaid
graph LR
    subgraph MPC["MPC Cycle"]
        direction LR
        A1["estimate<br/>state x(t)"] --> A2["predict<br/>trajectory"]
        A2 --> A3["optimize<br/>cost J"]
        A3 --> A4["apply first<br/>action u*(0)"]
        A4 --> A5["shift<br/>horizon"]
        A5 --> A1
    end

    subgraph Agent["Agent Cycle"]
        direction LR
        B1["normalize<br/>context"] --> B2["LLM<br/>inference"]
        B2 --> B3["detect<br/>tool call"]
        B3 --> B4["execute<br/>tool"]
        B4 --> B5["recurse<br/>with result"]
        B5 --> B1
    end

    classDef input fill:none,stroke:#60a5fa,stroke-width:2px
    classDef highlight fill:none,stroke:#f472b6,stroke-width:2px
    classDef output fill:none,stroke:#34d399,stroke-width:2px
    classDef result fill:none,stroke:#a78bfa,stroke-width:2px
    classDef progress fill:none,stroke:#fbbf24,stroke-width:2px

    class A1,B1 input
    class A2,B2 progress
    class A3,B3 highlight
    class A4,B4 output
    class A5,B5 result
```

The structure is isomorphic. The agent only returns control to the user when it decides the objective is met or when it hits a compute budget (the `maxTurns` cap -- analogous to a hard constraint on total actuation effort). Between those endpoints, it's a closed-loop controller running autonomously.

### The Bellman Connection

This loop is doing something deeper than it first appears. In optimal control, the Bellman equation defines the best possible cost-to-go from any state:

$$
V^*(x) = \min_u \Big[\ell(x, u) + V^*\big(f(x, u)\big)\Big]
$$

This says: the optimal value of being in state $x$ equals the cost of the best immediate action plus the optimal value of wherever that action takes you. It's recursive all the way down.

MPC *approximates* this by rolling the dynamics forward $N$ steps and optimizing over that finite window. Pure RL tries to learn $V^\ast$ globally from data. The agentic loop does a bit of both: the LLM's pre-training gives it an implicit $V^\ast$ (it "knows" what good code looks like), while the receding-horizon structure gives it the MPC-style finite lookahead to course-correct in real time.

### Devin's State-Passing Planner

Devin (and its open-source counterpart OpenHands) takes the MPC analogy even more literally. At every iteration, a planner function evaluates the current state vector -- working directory, open files, remaining step budget -- and decides whether to query the model, invoke a tool, or terminate. This is an explicit discrete-time MPC controller, with the state vector defined as a struct rather than a context window.

### The Ralph Wiggum Loop: Degenerate MPC

And then there's the Ralph Wiggum loop -- the stubbornly persistent pattern where you feed a fresh agent instance the same static prompt, let it try, check if the tests pass, and if not, feed the error logs to a *new* fresh instance. No memory. No prediction. No planning horizon. Just brute-force persistence verified by deterministic tests.

In control theory terms, this is a degenerate MPC with $N = 0$: zero prediction horizon, no internal model, pure reactive feedback. It works surprisingly well for tasks with strong external verifiers (compilers, test suites) because those verifiers act as a perfect cost function. But without a prediction horizon, it has no formal convergence guarantee. It's the control-theoretic equivalent of closing your eyes and repeatedly stomping the gas pedal until you're pointed in the right direction. Sometimes you get there. Sometimes you spin.

## Context Folding = Moving Horizon Estimation

Every Transformer-based LLM has a hard context limit. As the working history fills with code, error logs, and intermediate reasoning, performance degrades -- a phenomenon known as "context rot." It's the informational equivalent of mechanical friction: the more history you drag along, the harder it is to move forward.

In classical control, the dual problem to MPC is Moving Horizon Estimation. Instead of looking *forward* to plan optimal actions, MHE looks *backward* over a sliding window to estimate the current state from noisy observations. The math:

$$
\hat{x}_t = \arg\min_{x_{t-M:t}} \left[ \Gamma(x_{t-M}) + \sum_{k=t-M}^{t} \|y_k - h(x_k)\|^2 \right]
$$

Here $M$ is the estimation window and $\Gamma(x_{t-M})$ is the "arrival cost" -- a compressed summary of everything that happened *before* the window. You don't keep the full history. You keep a good-enough prior and a sliding window of recent data.

Context folding in agent systems does exactly this. Claude Code monitors the active token count. When it hits roughly 13,000 tokens below the context limit, it triggers a compaction protocol: the model summarizes the entire conversation history into a dense summary, preserving critical state (file modifications, active directory, unresolved tasks) while discarding the verbose intermediate steps.

A more sophisticated version is literal context folding: the agent branches into an isolated sub-trajectory to handle a localized subtask (fix a bug, install a dependency), then "folds" the entire branch into a single-sentence summary that gets injected back into the main planning thread. The main thread never sees the 50 tool calls it took to resolve a broken import -- it just sees "resolved dependency conflict in package X."

This is the arrival cost $\Gamma$. You don't need a perfect transcript of everything that happened. You need a *good enough prior* so the sliding window of recent context can do its job. Research on context folding shows it can reduce the active context footprint by 10x while matching or exceeding the performance of agents that try to keep everything in memory.

## When the Controller Fails

Here's where control theory really earns its keep. When you wrap an LLM in a recursive loop, it stops being a function and becomes a *discrete dynamical system* evolving through semantic space. And dynamical systems can exhibit distinct failure modes that map cleanly from classical control theory.

### Three Regimes

**Convergent (stable):** The agent systematically narrows toward the solution. Each iteration reduces the semantic distance to the goal. The diff gets smaller. The error count drops. The agent converges to a fixed point -- working code that passes all tests. This is the happy path: contractive dynamics.

**Oscillatory (underdamped):** The agent toggles between two competing fixes. It patches function A, which breaks test B. It patches test B, which re-breaks function A. The system cycles between two attractors without ever settling. In control theory, this is an underdamped oscillator -- there's a restoring force, but it overshoots on every cycle.

**Divergent (unstable):** The agent's attempts to fix errors introduce *more* errors. Each iteration pushes the state further from the goal. Hallucinated imports, phantom APIs, code that addresses errors which don't exist. This is the "hallucination spiral" -- an unstable system with no Lyapunov guarantee, diverging exponentially from the truth.

<div id="phase-portrait" style="text-align:center; margin: 2em 0;"></div>

<script>
(function() {
  'use strict';

  function render() {
    var container = d3.select('#phase-portrait');
    container.selectAll('*').remove();
    var c = ControlCharts.getColors();

    var width = 600, height = 420;
    var margin = { top: 36, right: 20, bottom: 50, left: 20 };
    var innerW = width - margin.left - margin.right;
    var innerH = height - margin.top - margin.bottom;
    var cx = margin.left + innerW / 2;
    var cy = margin.top + innerH / 2;

    var svg = container.append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .attr('width', '100%')
      .style('max-width', width + 'px')
      .style('display', 'block')
      .style('margin', '0 auto');

    // Background
    svg.append('rect')
      .attr('width', width).attr('height', height)
      .attr('fill', c.bg).attr('rx', 8);

    var g = svg.append('g');

    // Light grid
    var gridScale = d3.scaleLinear().domain([-3, 3]).range([margin.left, margin.left + innerW]);
    var gridScaleY = d3.scaleLinear().domain([-3, 3]).range([margin.top + innerH, margin.top]);
    [-2, -1, 0, 1, 2].forEach(function(v) {
      g.append('line')
        .attr('x1', gridScale(v)).attr('x2', gridScale(v))
        .attr('y1', margin.top).attr('y2', margin.top + innerH)
        .attr('stroke', c.grid).attr('stroke-width', 0.5);
      g.append('line')
        .attr('x1', margin.left).attr('x2', margin.left + innerW)
        .attr('y1', gridScaleY(v)).attr('y2', gridScaleY(v))
        .attr('stroke', c.grid).attr('stroke-width', 0.5);
    });

    // Axis lines
    g.append('line')
      .attr('x1', margin.left).attr('x2', margin.left + innerW)
      .attr('y1', cy).attr('y2', cy)
      .attr('stroke', c.muted).attr('stroke-width', 1);
    g.append('line')
      .attr('x1', cx).attr('x2', cx)
      .attr('y1', margin.top).attr('y2', margin.top + innerH)
      .attr('stroke', c.muted).attr('stroke-width', 1);

    // Axis labels
    g.append('text').attr('x', margin.left + innerW - 4).attr('y', cy - 8)
      .attr('text-anchor', 'end').attr('fill', c.muted).attr('font-size', '12px').text('x\u2081');
    g.append('text').attr('x', cx + 10).attr('y', margin.top + 14)
      .attr('text-anchor', 'start').attr('fill', c.muted).attr('font-size', '12px').text('x\u2082');

    // Fixed point marker at origin
    g.append('circle')
      .attr('cx', cx).attr('cy', cy).attr('r', 4)
      .attr('fill', c.attractor).attr('stroke', c.text).attr('stroke-width', 1);
    g.append('text').attr('x', cx + 8).attr('y', cy + 16)
      .attr('fill', c.attractor).attr('font-size', '11px').text('equilibrium');

    // Generate trajectories
    var scale = Math.min(innerW, innerH) / 2 * 0.85;

    function spiralPoints(r0, decay, nTurns, nPoints) {
      var pts = [];
      for (var i = 0; i <= nPoints; i++) {
        var t = i / nPoints * nTurns * 2 * Math.PI;
        var r = r0 * Math.exp(decay * t);
        pts.push([cx + r * Math.cos(t) * scale, cy - r * Math.sin(t) * scale]);
      }
      return pts;
    }

    // Convergent: damped spiral inward
    var convergentPts = spiralPoints(0.9, -0.12, 3, 200);
    // Oscillatory: stable limit cycle (circle)
    var oscPts = [];
    var oscR = 0.5;
    for (var i = 0; i <= 200; i++) {
      var t = i / 200 * 3 * 2 * Math.PI;
      oscPts.push([cx + oscR * Math.cos(t) * scale, cy - oscR * Math.sin(t) * scale]);
    }
    // Divergent: expanding spiral outward
    var divergentPts = spiralPoints(0.15, 0.09, 3, 200);

    var line = d3.line().curve(d3.curveCatmullRom);

    var trajectories = [
      { pts: convergentPts, color: c.convergent, label: 'Convergent' },
      { pts: oscPts, color: c.oscillatory, label: 'Oscillatory' },
      { pts: divergentPts, color: c.divergent, label: 'Divergent' },
    ];

    trajectories.forEach(function(traj) {
      var path = g.append('path')
        .datum(traj.pts)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', traj.color)
        .attr('stroke-width', 2.5)
        .attr('stroke-linecap', 'round');

      var totalLen = path.node().getTotalLength();
      path
        .attr('stroke-dasharray', totalLen)
        .attr('stroke-dashoffset', totalLen)
        .transition()
        .duration(3000)
        .ease(d3.easeLinear)
        .attr('stroke-dashoffset', 0);

      // Arrowhead at the end
      var endPt = traj.pts[traj.pts.length - 1];
      var prevPt = traj.pts[traj.pts.length - 5] || traj.pts[traj.pts.length - 2];
      var angle = Math.atan2(endPt[1] - prevPt[1], endPt[0] - prevPt[0]) * 180 / Math.PI;
      g.append('polygon')
        .attr('points', '0,-4 10,0 0,4')
        .attr('fill', traj.color)
        .attr('transform', 'translate(' + endPt[0] + ',' + endPt[1] + ') rotate(' + angle + ')')
        .attr('opacity', 0)
        .transition().delay(3000).duration(200).attr('opacity', 1);

      // Starting dot
      g.append('circle')
        .attr('cx', traj.pts[0][0]).attr('cy', traj.pts[0][1]).attr('r', 3.5)
        .attr('fill', traj.color).attr('opacity', 0.7);
    });

    // Legend
    var legendX = width / 2 - 140;
    var legendY = 16;
    trajectories.forEach(function(traj, i) {
      svg.append('line')
        .attr('x1', legendX + i * 110).attr('x2', legendX + i * 110 + 18)
        .attr('y1', legendY).attr('y2', legendY)
        .attr('stroke', traj.color).attr('stroke-width', 2.5);
      svg.append('text')
        .attr('x', legendX + i * 110 + 24).attr('y', legendY + 4)
        .attr('fill', c.text).attr('font-size', '12px').text(traj.label);
    });

    // Replay button
    var btnG = svg.append('g')
      .attr('transform', 'translate(' + (width - 70) + ',' + (height - 24) + ')')
      .style('cursor', 'pointer')
      .on('click', function() { render(); });
    btnG.append('rect')
      .attr('width', 56).attr('height', 22).attr('rx', 4)
      .attr('fill', 'none').attr('stroke', c.muted).attr('stroke-width', 1);
    btnG.append('text')
      .attr('x', 28).attr('y', 15).attr('text-anchor', 'middle')
      .attr('fill', c.muted).attr('font-size', '11px').text('Replay');
  }

  render();
  document.documentElement.addEventListener('themechange', render);
})();
</script>

### The Lyapunov Lens

In classical control, you prove stability by finding a Lyapunov function $V(x)$ -- a scalar measure of "energy" in the system -- and showing it decreases at every step:

$$
\Delta V(x) = V(x_{k+1}) - V(x_k) < 0
$$

If you can prove $\Delta V < 0$ everywhere, the system is guaranteed to converge to the equilibrium. No exceptions.

For an AI agent, think of $V(x)$ as the "semantic distance" between the current codebase state and the goal state. If every agent action reduces this distance -- fewer failing tests, smaller diffs, cleaner error logs -- you have a stable system. The moment $\Delta V$ goes positive (an edit that introduces more problems than it solves), you've lost your stability guarantee.

This framing explains *why* the Ralph Wiggum loop sometimes works: the external test suite acts as a perfect Lyapunov function. Each iteration either passes (distance drops to zero) or fails (distance stays positive, but the fresh context prevents *accumulation* of errors). The statelessness is a feature -- it prevents the divergent spiral by resetting the system state each iteration, at the cost of discarding all momentum.

It also explains the "epistemic speed limit": if you iterate too fast without sufficient planning depth, minor semantic errors accumulate faster than the feedback loop can correct them. The agent enters a regime of *agentic collapse* -- iterating furiously, accomplishing nothing. MPC's prediction horizon is the antidote: think before you act, and *only then* commit.

## Computational Physics

There's a deeper pattern here that I find genuinely beautiful. MPC was invented to manage the physics of chemical reactors -- momentum, inertia, friction, disturbances. Context rot *is* friction, degrading performance over time. Token generation *is* momentum, driving the system forward along a vector that's hard to arrest. Bugs and missing dependencies *are* external disturbances, randomly deflecting the planned trajectory.

An AI agent navigating a million-line undocumented codebase to implement a new feature is mathematically indistinguishable from a drone navigating a dense, unpredictable forest. The trajectory is semantic rather than spatial, but the optimization problem is identical: minimize cost, respect constraints, re-plan continuously, don't crash.

The future of AI agents isn't prompt engineering -- it's control engineering. The frameworks that will win aren't the ones with the cleverest system prompts. They're the ones with the tightest feedback loops, the deepest prediction horizons, and the most rigorous guarantees that $\Delta V < 0$ at every step.

![TL;DR -- Your AI agent is a receding-horizon controller: it observes state, predicts outcomes, picks the best next action, re-evaluates. Context folding is moving horizon estimation, failure modes are Lyapunov stability regimes, and the future of agents is control engineering, not prompt engineering.](/assets/posts-media/claude-code-control-theory.jpg)

---

## References

1. "Model predictive control," Wikipedia. [Link](https://en.wikipedia.org/wiki/Model_predictive_control)
2. J. B. Rawlings, D. Q. Mayne, and M. M. Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed. Nob Hill Publishing, 2017. [PDF](https://sites.engineering.ucsb.edu/~jbraw/mpc/MPC-book-2nd-edition-1st-printing.pdf)
3. "LLM-Based World Models," Emergent Mind. [Link](https://www.emergentmind.com/topics/llm-based-world-models)
4. Y. Feng et al., "Scaling Long-Horizon LLM Agent via Context-Folding," arXiv:2510.11967, 2025. [Link](https://arxiv.org/abs/2510.11967)
5. "Claude Code: Behind-the-scenes of the master agent loop," PromptLayer Blog. [Link](https://blog.promptlayer.com/claude-code-behind-the-scenes-of-the-master-agent-loop/)
6. "Effective context engineering for AI agents," Anthropic. [Link](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
7. G. L. Bailo, "Why AI Agents Fail: The Stochastic Convergence Spiral," Medium, 2025. [Link](https://medium.com/@gianlucabailo/why-ai-agents-fail-the-stochastic-convergence-spiral-4ab5a8aa0ef4)
8. Z. Chen et al., "Geometric Dynamics of Agentic Loops in Large Language Models," arXiv:2512.10350, 2025. [Link](https://arxiv.org/abs/2512.10350)
9. H. Kim et al., "Test-Time Alignment for Large Language Models via Textual Model Predictive Control," arXiv:2502.20795, 2025. [Link](https://arxiv.org/abs/2502.20795)
10. "Multi-Dimensional Constraint Integration Method for Large Language Models via Lyapunov Stability Theory," OpenReview, 2025. [Link](https://openreview.net/forum?id=rbl8fHjLuF)
11. "Agentic Collapse: A Time-Delayed Cybernetic Framework for Epistemic Stability in Autonomous AI Systems," ResearchGate, 2025. [Link](https://www.researchgate.net/publication/399368003)
12. A. Gekov, "2026 -- The year of the Ralph Loop Agent," DEV Community. [Link](https://dev.to/alexandergekov/2026-the-year-of-the-ralph-loop-agent-1gkj)
13. "Ralph Wiggum Loop," beuke.org. [Link](https://beuke.org/ralph-wiggum-loop/)
14. F. Wang et al., "When control meets large language models: From words to dynamics," arXiv:2602.03433, 2026. [Link](https://arxiv.org/html/2602.03433v1)
15. "Recursive Language Models: the paradigm of 2026," Prime Intellect. [Link](https://www.primeintellect.ai/blog/rlm)
