---
layout: post
title: "Your AI Agent Is a Control System (It Just Doesn't Know It Yet)"
date: 2026-02-26 10:00:00 -0700
tags: ai llm control-theory machine-learning
author: bolu-atx
categories: machine-learning
---

Suppose your coding agent sees three failing tests, opens the wrong file, makes an edit that fixes one failure and creates two new ones, runs the tests again, notices the blast radius, backs up, and tries a narrower patch.

That is not "just inference." That's a feedback loop.

More specifically: it's an iterative policy acting on a partially observed environment, using fresh observations to update its next move. If you come from ML, that's already enough to make the control-theory comparison useful. You don't need to believe an LLM agent is literally an industrial controller. You just need to notice that once the model is embedded in a tool-use loop, the thing you're evaluating is no longer a one-shot predictor. It's a dynamical system.

That shift matters because it changes what "good" means.

- A good base model is not automatically a good closed-loop agent.
- A bad planner can destabilize a strong model.
- A weak verifier can make a bad agent look competent for a surprisingly long time.
- Most real failures are not "wrong answer once." They're oscillation, drift, and local hacks that look good for two steps and bad for twenty.

That is the part I think control gives us: not fancy vocabulary, but a cleaner way to talk about what these systems are doing, where they fail, and what to optimize.

<!--more-->

## The Useful Claim, Stripped Down

Here's the claim in plain English:

> An agent is a controller wrapped around a world model and a set of actuators.

The "world model" is the LLM's learned prior over how code, shells, APIs, and people behave. The "actuators" are tool calls: edit a file, run a command, open a browser tab, send a message, call a retriever. The "plant" is the external environment those actions hit: the repo, the compiler, the test suite, the network, the human user.

Once you wire those pieces together, the loop looks like this:

1. Observe the current state as imperfectly as you can.
2. Predict what a few candidate actions will do.
3. Pick one action.
4. Execute it in the real world.
5. Measure the result.
6. Repeat until you hit the goal, a budget, or a failure condition.

That is the control lens. Not "LLMs secretly solve Riccati equations." Just: this is a closed-loop sequential decision system, so the right questions are now about feedback, stability, observability, horizon length, and cost shaping.

## The Mapping That Actually Matters

You can map the usual control terms onto agent systems pretty directly:

| Control idea | Agent analogue | Why it matters |
|:---|:---|:---|
| Plant | External environment | Repo, shell, browser, APIs, user |
| State | Working state | Files read, diffs applied, errors observed, plan status |
| Observation | Context | What the model can currently see about that state |
| Controller | Agent policy | The logic that chooses the next action |
| Action | Tool call | Edit, search, test, browse, ask, terminate |
| Dynamics | Environment response | What actually happens after the action |
| Cost | Reward / penalties | Time, tokens, failures, broken tests, user dissatisfaction |
| Horizon | Planning depth | How far ahead the agent reasons before acting |

The important caveat is that this mapping is not exact. Classical control usually assumes a cleaner state representation, known dynamics, and explicit costs. Agents have none of that. Their state is messy, partially latent, and partly summarized in natural language. Their dynamics are a mix of code semantics, tool behavior, and model beliefs. Their cost function is often implicit and badly specified.

But "messy control problem" is still a control problem.

## Why Receding Horizon Is the Right Mental Model

The most useful control analogy here is model predictive control, or MPC.

In MPC, you simulate a few steps ahead, pick the best action sequence, execute only the first action, then re-plan from the new measured state. That last part is the whole game. You don't commit to your full rollout because the world is going to surprise you.

That is exactly how good agents behave.

They do not decide on a 20-step plan and then march through it blindly. They sketch a trajectory, take one action, look at the new evidence, and update. If the test output or file contents disagree with the model's prediction, the plan should change immediately.

The standard finite-horizon objective is:

$$
\min_{u_0, \ldots, u_{N-1}} \sum_{k=0}^{N-1} \ell(x_k, u_k) + V_f(x_N)
$$

subject to

$$
x_{k+1} = f(x_k, u_k)
$$

In agent terms, that says: choose a sequence of candidate actions that trades off short-term cost against long-term task completion, given your current guess about how the environment will respond.

The practical translation is less glamorous:

- don't optimize only for the next token
- don't optimize only for the final answer
- optimize for trajectories

That means the unit of evaluation is not "was this thought good?" It's "did this loop move the system toward the goal?"

## A Coding Agent Really Does Run a Receding-Horizon Loop

If you look at real coding agents, the loop has the same shape:

1. compress or normalize state
2. infer the next move
3. detect whether that move needs a tool
4. execute the tool
5. feed the result back into the next iteration

That is not a metaphor in the hand-wavy sense. The implementation details differ, but architecturally it's the same receding-horizon pattern: estimate, act, observe, re-plan.

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

    var totalSteps = 20;
    var horizonLen = 5;
    var actual = [1.0];
    var rng = d3.randomNormal(0, 0.04);
    for (var i = 1; i <= totalSteps; i++) {
      var prev = actual[i - 1];
      var decay = prev * 0.92 + rng();
      if (i === 6) decay = prev + 0.18;
      if (i === 14) decay = prev + 0.14;
      actual.push(Math.max(0.02, Math.min(1.2, decay)));
    }

    var x = d3.scaleLinear().domain([0, totalSteps]).range([0, innerW]);
    var y = d3.scaleLinear().domain([0, 1.25]).range([innerH, 0]);

    y.ticks(5).forEach(function(v) {
      g.append('line')
        .attr('x1', 0).attr('x2', innerW)
        .attr('y1', y(v)).attr('y2', y(v))
        .attr('stroke', c.grid).attr('stroke-width', 0.5);
    });

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

    svg.append('text')
      .attr('x', margin.left + innerW / 2).attr('y', height - 6)
      .attr('text-anchor', 'middle').attr('fill', c.text).attr('font-size', '12px')
      .text('Agent step');
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(margin.top + innerH / 2)).attr('y', 14)
      .attr('text-anchor', 'middle').attr('fill', c.text).attr('font-size', '12px')
      .text('Distance to goal proxy');

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
      .x(function(d) { return x(d[0]); })
      .y(function(d) { return y(d[1]); })
      .curve(d3.curveMonotoneX);

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

    var currentStep = 0;

    function drawStep() {
      if (currentStep > totalSteps) {
        clearInterval(animTimer);
        animTimer = null;
        return;
      }

      var actualData = [];
      for (var i = 0; i <= currentStep; i++) {
        actualData.push([i, actual[i]]);
      }
      actualPath.attr('d', line(actualData));

      stepDot.attr('cx', x(currentStep)).attr('cy', y(actual[currentStep]));
      stepLabel.attr('x', x(currentStep)).attr('y', y(actual[currentStep]) - 12)
        .text('step ' + currentStep);

      var hStart = currentStep;
      var hEnd = Math.min(currentStep + horizonLen, totalSteps);
      horizonRect
        .attr('x', x(hStart)).attr('y', 0)
        .attr('width', x(hEnd) - x(hStart))
        .attr('height', innerH);

      if (currentStep < totalSteps) {
        var plannedData = [];
        var pVal = actual[currentStep];
        for (var j = currentStep; j <= hEnd; j++) {
          plannedData.push([j, pVal]);
          pVal = pVal * 0.88;
        }
        plannedPath.attr('d', line(plannedData));
      }

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
        B1["summarize<br/>state"] --> B2["infer next<br/>move"]
        B2 --> B3["select<br/>tool"]
        B3 --> B4["execute<br/>tool"]
        B4 --> B5["update plan<br/>with result"]
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

What does that buy you?

For me, three things:

1. It tells you why single-step evals are not enough. A model can look great at choosing the next action and still be terrible over 30 steps because its errors compound.
2. It tells you why verifiers matter so much. Fast, high-signal feedback is what keeps the loop from drifting.
3. It tells you why planning depth is a resource. Too short and the agent thrashes. Too long and it hallucinates a brittle plan that reality invalidates immediately.

## Context Management Is State Estimation

The second useful control analogy is state estimation.

An agent never sees the full environment directly. It sees shell output, diffs, logs, snippets, summaries, and maybe the user's last message. That is an observation stream, not ground truth. The agent has to compress those observations into an internal state that is good enough for the next action.

That is why context management matters so much. Summaries are not just token-budget hacks. They are your state estimator.

In classical control, moving horizon estimation keeps a recent window of observations plus a compact summary of older history. Agent systems do the same thing when they summarize prior steps, fold resolved subtasks into a short note, or carry forward only the facts that still constrain the future.

When that summary is bad, the controller acts on the wrong state. You see this immediately in practice:

- the agent forgets which file it already modified
- it reopens a bug it had already fixed
- it keeps debugging an error that no longer exists
- it starts planning around stale assumptions from 20 turns ago

That is not just "context window pressure." It is a state-estimation failure.

This is also where a lot of current agent work feels more like systems engineering than pure modeling. Better summarization, better scratchpads, better memory schemas, better subtask boundaries, better retrieval, better tool output formatting: all of these improve the quality of the state estimate the policy is acting on.

## Failure Modes Look Like Control Problems Too

Once you view the loop as a dynamical system, the common failure modes stop looking random.

### Convergent

The agent makes progress monotonically enough that small mistakes get corrected and the loop still settles. Test failures trend down. Diffs get narrower. The agent's search becomes more local as it approaches the target.

### Oscillatory

The agent flips between incompatible local fixes.

You see this when it alternates between two hypotheses:

- patch implementation to satisfy test A
- patch test fixture to satisfy implementation
- revert patch because test B now fails
- reintroduce the original behavior because test A broke again

This is the agent equivalent of an underdamped system. There is feedback, but the gain is wrong or the state estimate is incomplete, so it overshoots.

### Divergent

The loop gets worse with each iteration. New edits create more failures than they remove. The agent starts reasoning from artifacts it introduced itself. It chases phantom APIs, stale stack traces, or nonexistent invariants.

That is the failure mode people often call "hallucination," but "divergence" is more precise. The issue is not just that one belief is false. The issue is that the closed-loop system is moving away from the target.

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

    svg.append('rect')
      .attr('width', width).attr('height', height)
      .attr('fill', c.bg).attr('rx', 8);

    var g = svg.append('g');

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

    g.append('line')
      .attr('x1', margin.left).attr('x2', margin.left + innerW)
      .attr('y1', cy).attr('y2', cy)
      .attr('stroke', c.muted).attr('stroke-width', 1);
    g.append('line')
      .attr('x1', cx).attr('x2', cx)
      .attr('y1', margin.top).attr('y2', margin.top + innerH)
      .attr('stroke', c.muted).attr('stroke-width', 1);

    g.append('text').attr('x', margin.left + innerW - 4).attr('y', cy - 8)
      .attr('text-anchor', 'end').attr('fill', c.muted).attr('font-size', '12px').text('x\u2081');
    g.append('text').attr('x', cx + 10).attr('y', margin.top + 14)
      .attr('text-anchor', 'start').attr('fill', c.muted).attr('font-size', '12px').text('x\u2082');

    g.append('circle')
      .attr('cx', cx).attr('cy', cy).attr('r', 4)
      .attr('fill', c.attractor).attr('stroke', c.text).attr('stroke-width', 1);
    g.append('text').attr('x', cx + 8).attr('y', cy + 16)
      .attr('fill', c.attractor).attr('font-size', '11px').text('equilibrium');

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

    var convergentPts = spiralPoints(0.9, -0.12, 3, 200);
    var oscPts = [];
    var oscR = 0.5;
    for (var i = 0; i <= 200; i++) {
      var t = i / 200 * 3 * 2 * Math.PI;
      oscPts.push([cx + oscR * Math.cos(t) * scale, cy - oscR * Math.sin(t) * scale]);
    }
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

      var endPt = traj.pts[traj.pts.length - 1];
      var prevPt = traj.pts[traj.pts.length - 5] || traj.pts[traj.pts.length - 2];
      var angle = Math.atan2(endPt[1] - prevPt[1], endPt[0] - prevPt[0]) * 180 / Math.PI;
      g.append('polygon')
        .attr('points', '0,-4 10,0 0,4')
        .attr('fill', traj.color)
        .attr('transform', 'translate(' + endPt[0] + ',' + endPt[1] + ') rotate(' + angle + ')')
        .attr('opacity', 0)
        .transition().delay(3000).duration(200).attr('opacity', 1);

      g.append('circle')
        .attr('cx', traj.pts[0][0]).attr('cy', traj.pts[0][1]).attr('r', 3.5)
        .attr('fill', traj.color).attr('opacity', 0.7);
    });

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

## You Probably Can't Prove Stability, But You Can Instrument It

Classical control has Lyapunov functions: scalar quantities that go down when the system is moving toward a stable equilibrium.

For agents, we usually do not have anything that clean. There is no general scalar "distance to solved" for arbitrary software tasks.

But the instinct is still right. You want proxies for progress, and you want to know when they stop improving.

For coding agents, useful progress signals often look like:

- number of failing tests
- compile errors
- linter violations
- diff size
- number of files touched
- repeated edits to the same region
- repeated execution of the same command without new information

None of these is a perfect Lyapunov function. All of them can be gamed. But if you log them over time, you can tell the difference between progress, dithering, and collapse much earlier than if you only inspect the final answer.

This is the practical "what does it mean?" part:

- Treat agent runs as trajectories, not outputs.
- Instrument intermediate state, not just final success.
- Add guardrails when progress proxies flatten or reverse.
- Terminate or escalate when the loop is clearly oscillating.

That is a control mindset.

## What This Changes for Agent Builders

If the control analogy is useful, it should change design decisions. I think it changes at least four.

### 1. Optimize the feedback loop, not just the model

If your verifier is slow, sparse, or ambiguous, your agent is flying mostly open-loop.

A lot of "agent capability" is really feedback quality:

- fast tests beat slow tests
- structured tool output beats raw terminal spam
- narrow diffs beat repo-wide rewrites
- explicit constraints beat vague instructions

A stronger base model helps. But a mediocre model with tight feedback can outperform a stronger model with weak feedback.

### 2. Make state legible

The policy can only act on the state representation you give it.

So make that state cheap to inspect and easy to summarize:

- expose plan state explicitly
- record what has already been tried
- track edited files and unresolved failures
- format tool results so important deltas are obvious

If the agent keeps forgetting what happened, don't only blame the context window. Blame the state representation.

### 3. Control the action space

Controllers get easier to stabilize when the actuators are sensible.

The same is true here. Agents behave better when the available actions are high-signal and constrained:

- use purpose-built tools instead of raw shell when possible
- prefer patch tools over unconstrained file rewrites
- separate "inspect" actions from "mutate" actions
- require verification after high-impact edits

Part of agent engineering is actuator design.

### 4. Measure horizon quality

Long-horizon capability is not "can the model describe a long plan?" It is "can the loop stay coherent while repeatedly updating that plan under feedback?"

Those are very different abilities.

This is why agent benchmarks that grade only final answers miss something essential. Two systems may both finish the task, but one may do it with tight convergence and the other with catastrophic instability that happened to get lucky before the budget ran out.

## The Real Payoff

So what does it mean to say "AI agents are control systems"?

For me, it means four concrete things:

1. Stop thinking of an agent as a single prediction.
2. Start thinking of it as a closed-loop process with memory, actions, observations, and failure modes.
3. Evaluate the trajectory, not just the endpoint.
4. Design the loop so that reality corrects the model quickly and cheaply.

That's it.

The control framing is useful not because it makes agents sound more rigorous than they are. It's useful because it forces a more honest question:

When this thing acts repeatedly on the world, under imperfect information, with limited compute and noisy feedback, does it settle toward the goal or not?

That is the question control theory has been asking for decades. We should probably ask it more often in agent engineering too.

![TL;DR -- Once an LLM is embedded in a tool-use loop, you're no longer evaluating a one-shot predictor. You're evaluating a closed-loop system. The useful control questions are about feedback quality, state estimation, horizon length, and stability under repeated action.](/assets/posts-media/claude-code-control-theory.jpg)

---

## References

1. "Model predictive control," Wikipedia. [Link](https://en.wikipedia.org/wiki/Model_predictive_control)
2. J. B. Rawlings, D. Q. Mayne, and M. M. Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed. Nob Hill Publishing, 2017. [PDF](https://sites.engineering.ucsb.edu/~jbraw/mpc/MPC-book-2nd-edition-1st-printing.pdf)
3. "Effective context engineering for AI agents," Anthropic. [Link](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
4. Y. Feng et al., "Scaling Long-Horizon LLM Agent via Context-Folding," arXiv:2510.11967, 2025. [Link](https://arxiv.org/abs/2510.11967)
5. H. Kim et al., "Test-Time Alignment for Large Language Models via Textual Model Predictive Control," arXiv:2502.20795, 2025. [Link](https://arxiv.org/abs/2502.20795)
6. F. Wang et al., "When control meets large language models: From words to dynamics," arXiv:2602.03433, 2026. [Link](https://arxiv.org/html/2602.03433v1)
