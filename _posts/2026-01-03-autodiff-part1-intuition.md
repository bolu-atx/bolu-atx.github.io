---
layout: post
title:  "Understanding Autodiff, Part 1: Thinking in Graphs"
date:   2026-01-03 09:00:00 -0700
tags: rust machine-learning programming
author: bolu-atx
categories: programming
---

I've always found automatic differentiation a bit magical. You write some math, call `.backward()`, and somehow the computer figures out all the derivatives. For years I used it without really understanding it.

Back in grad school, when I was writing DAE/ODE solvers using RK45 integrators, Professor Michael Baldea would mention that automatic differentiation was the "hot new research area." It sounded like magic to me at the time. Fast forward to today, and it's everywhere — PyTorch, JAX, tinygrad, anywhere there's backprop and gradients. What was once cutting-edge research is now something we take for granted.

Then I sat down and implemented one from scratch. Turns out it's not magic at all — it's actually a beautiful idea that clicks once you see it the right way.

<!--more-->

## Forget Calculus Class (For Now)

When most of us learned calculus, we memorized rules:

- $\frac{d}{dx}(x^2) = 2x$
- $\frac{d}{dx}(\sin x) = \cos x$
- The chain rule: $\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$

These rules work great on paper. But here's the thing — computers don't manipulate symbols the way we do. They work with numbers and data structures.

So instead of asking "how do I differentiate this expression?", let's ask a different question:

**How do I represent a mathematical expression in a way that makes differentiation natural?**

## Math as a Picture

Consider this function:

$$f(x, y) = x \cdot y + \sin(x)$$

Nothing fancy — multiplication, addition, a sine. But let's draw it differently. Instead of one line of symbols, let's trace how the computation actually happens:

```mermaid
graph BT
    x["x"]
    y["y"]
    mul["multiply"]
    sin["sine"]
    add["add"]
    out["output"]

    x --> mul
    y --> mul
    x --> sin
    mul --> add
    sin --> add
    add --> out

    style x fill:none,stroke:#5a9fd4,stroke-width:2px
    style y fill:none,stroke:#5a9fd4,stroke-width:2px
    style out fill:none,stroke:#4CAF50,stroke-width:2px
```

Read this bottom-to-top. The inputs `x` and `y` flow upward. They get combined by operations. Eventually we get our output.

This is a **computation graph**. Every mathematical expression can be drawn this way.

Let me add the actual values. Say x = 2 and y = 3:

```mermaid
graph BT
    x["x = 2"]
    y["y = 3"]
    mul["multiply<br/>2 × 3 = 6"]
    sin["sine<br/>sin(2) ≈ 0.91"]
    add["add<br/>6 + 0.91 = 6.91"]

    x --> mul
    y --> mul
    x --> sin
    mul --> add
    sin --> add

    style x fill:none,stroke:#5a9fd4,stroke-width:2px
    style y fill:none,stroke:#5a9fd4,stroke-width:2px
    style add fill:none,stroke:#4CAF50,stroke-width:2px
```

Now you can literally *see* the computation happening. Values flow up from inputs to output.

## The Question We Actually Want to Answer

Here's what we care about in machine learning:

> If I wiggle the inputs a tiny bit, how much does the output wiggle?

That's what a derivative is. It's a measure of sensitivity.

For our function at $x=2$, $y=3$:
- If I increase $x$ slightly, how much does the output change? (That's $\partial f/\partial x$)
- If I increase $y$ slightly, how much does the output change? (That's $\partial f/\partial y$)

The graph view makes this intuitive. Wiggle x, and that wiggle propagates up through every node that depends on x.

## Following the Wiggle

Let's trace what happens when we wiggle $x$:

```mermaid
graph BT
    x["x = 2<br/>↑ wiggle here"]
    y["y = 3"]
    mul["multiply<br/>feels the wiggle"]
    sin["sine<br/>feels the wiggle"]
    add["add<br/>feels both wiggles"]

    x --> mul
    y --> mul
    x --> sin
    mul --> add
    sin --> add

    linkStyle 0 stroke:#ff6b6b,stroke-width:3px
    linkStyle 2 stroke:#ff6b6b,stroke-width:3px
    linkStyle 3 stroke:#ff6b6b,stroke-width:3px
    linkStyle 4 stroke:#ff6b6b,stroke-width:3px

    style x fill:none,stroke:#E91E63,stroke-width:2px
```

Notice something important: **$x$ connects to the output through two different paths**.

1. x → multiply → add → output
2. x → sine → add → output

Both paths carry the wiggle. The total effect on the output is the *sum* of both contributions.

This is the key insight. In a graph, derivatives flow backward along edges, and when paths merge, contributions add up.

## Local Rules, Global Result

Here's what makes autodiff elegant. Each operation only needs to know one thing: **how sensitive is my output to each of my inputs?**

Let's write these down:

| Operation | If input wiggles by $\varepsilon$... | Output wiggles by... |
|-----------|--------------------------|---------------------|
| $\text{add}(a, b)$ | $a$ wiggles by $\varepsilon$ | $\varepsilon$ (just passes through) |
| $\text{add}(a, b)$ | $b$ wiggles by $\varepsilon$ | $\varepsilon$ (just passes through) |
| $\text{multiply}(a, b)$ | $a$ wiggles by $\varepsilon$ | $b \cdot \varepsilon$ (scaled by the other input) |
| $\text{multiply}(a, b)$ | $b$ wiggles by $\varepsilon$ | $a \cdot \varepsilon$ (scaled by the other input) |
| $\sin(a)$ | $a$ wiggles by $\varepsilon$ | $\cos(a) \cdot \varepsilon$ |

These are just the derivatives you learned in calculus, but think of them as **local exchange rates** for wiggles.

## The Chain Rule as Plumbing

Now here's the beautiful part. To find the total sensitivity from input to output, we just:

1. Find all paths from input to output
2. Multiply the local exchange rates along each path
3. Add up the contributions from all paths

Let's do it for $\partial f/\partial x$:

**Path 1: x → multiply → add**
- $x \to \text{multiply}$: exchange rate $= y = 3$ (the other input to multiply)
- $\text{multiply} \to \text{add}$: exchange rate $= 1$ (addition just passes through)
- Total for this path: $3 \times 1 = 3$

**Path 2: x → sin → add**
- $x \to \sin$: exchange rate $= \cos(2) \approx -0.42$
- $\sin \to \text{add}$: exchange rate $= 1$
- Total for this path: $-0.42 \times 1 = -0.42$

**Grand total: $3 + (-0.42) = 2.58$**

That's $\partial f/\partial x$. We computed a derivative without doing any symbolic manipulation — just multiplying and adding numbers along paths in a graph.

```mermaid
graph BT
    x["x = 2"]
    y["y = 3"]
    mul["×<br/>local rate: y=3"]
    sin["sin<br/>local rate: cos(2)≈-0.42"]
    add["+<br/>local rate: 1, 1"]

    x -->|"×3"| mul
    y --> mul
    x -->|"×(-0.42)"| sin
    mul -->|"×1"| add
    sin -->|"×1"| add

    style x fill:none,stroke:#E91E63,stroke-width:2px
```

## Why Not Just Go Forward?

You might wonder: can't we just wiggle each input and measure what happens at the output?

Yes! That's called **forward mode** autodiff. For each input, you trace the wiggle forward through the graph.

But think about a neural network. It might have millions of input parameters. Forward mode would require millions of passes through the graph — one for each parameter.

Here's the trick: what if we went **backward** instead?

## Thinking Backward

Instead of asking "if I wiggle $x$, what happens to output?", ask:

> "If the output needed to change, how much would each input need to change?"

Start at the output. Its "sensitivity to itself" is 1 (trivially). Now work backward:

```mermaid
graph BT
    x["x<br/>sensitivity: ?"]
    y["y<br/>sensitivity: ?"]
    mul["×<br/>sensitivity: ?"]
    sin["sin<br/>sensitivity: ?"]
    add["+<br/>sensitivity: 1"]

    x --> mul
    y --> mul
    x --> sin
    mul --> add
    sin --> add

    style add fill:none,stroke:#4CAF50,stroke-width:2px
```

The add node has sensitivity 1. It passes this backward to both its inputs (because addition has local rate 1 in both directions):

```mermaid
graph BT
    x["x<br/>sensitivity: ?"]
    y["y<br/>sensitivity: ?"]
    mul["×<br/>sensitivity: 1"]
    sin["sin<br/>sensitivity: 1"]
    add["+<br/>sensitivity: 1 ✓"]

    x --> mul
    y --> mul
    x --> sin
    mul --> add
    sin --> add

    style add fill:none,stroke:#4CAF50,stroke-width:2px
    style mul fill:none,stroke:#f9a825,stroke-width:2px
    style sin fill:none,stroke:#f9a825,stroke-width:2px
```

Now multiply and sin both have sensitivity 1. They propagate backward using their local rates:

- From multiply: $x$ gets $1 \times y = 1 \times 3 = 3$, $y$ gets $1 \times x = 1 \times 2 = 2$
- From sin: $x$ gets $1 \times \cos(2) \approx -0.42$

But wait — $x$ receives from both multiply AND sin! We add them up:

```mermaid
graph BT
    x["x<br/>sensitivity: 3 + (-0.42) = 2.58 ✓"]
    y["y<br/>sensitivity: 2 ✓"]
    mul["×<br/>sensitivity: 1 ✓"]
    sin["sin<br/>sensitivity: 1 ✓"]
    add["+<br/>sensitivity: 1 ✓"]

    x --> mul
    y --> mul
    x --> sin
    mul --> add
    sin --> add

    style add fill:none,stroke:#4CAF50,stroke-width:2px
    style mul fill:none,stroke:#4CAF50,stroke-width:2px
    style sin fill:none,stroke:#4CAF50,stroke-width:2px
    style x fill:none,stroke:#9C27B0,stroke-width:2px
    style y fill:none,stroke:#9C27B0,stroke-width:2px
```

**One backward pass gave us *all* the derivatives.**

This is reverse-mode autodiff. It's why PyTorch can train a model with 175 billion parameters — one forward pass to compute the loss, one backward pass to get all 175 billion gradients.

## The Mental Model

Here's how I think about it now:

**Forward pass**: Values flow upward like water. Inputs combine, transform, eventually reach the output.

**Backward pass**: Sensitivity flows downward like... inverse water? Each node receives sensitivity from above, multiplies by its local rate, and passes it down.

```mermaid
graph BT
    subgraph "Forward: Values Flow Up"
        direction BT
        i1["inputs"] --> o1["operations"] --> out1["output"]
    end

    subgraph "Backward: Sensitivity Flows Down"
        direction TB
        out2["output<br/>sens = 1"] --> o2["operations"] --> i2["inputs<br/>sens = gradients!"]
    end
```

When paths split going forward, sensitivities add going backward. When paths merge going forward, sensitivities... well, they just follow their edges backward.

## What About More Complex Graphs?

The same principle scales to any computation:

```mermaid
graph BT
    x["x"]
    y["y"]
    a["a = x + y"]
    b["b = x - y"]
    c["c = a × b"]

    x --> a
    y --> a
    x --> b
    y --> b
    a --> c
    b --> c

    style x fill:none,stroke:#5a9fd4,stroke-width:2px
    style y fill:none,stroke:#5a9fd4,stroke-width:2px
    style c fill:none,stroke:#4CAF50,stroke-width:2px
```

This is $c = (x + y)(x - y) = x^2 - y^2$. The graph has a diamond shape — $x$ and $y$ each flow through two different paths before rejoining.

Backward pass still works the same way:
1. Start at $c$ with sensitivity 1
2. $c$ passes to $a$: sensitivity $\times$ (value of $b$)
3. $c$ passes to $b$: sensitivity $\times$ (value of $a$)
4. $a$ and $b$ pass to $x$ and $y$, with appropriate signs
5. $x$ and $y$ sum their incoming sensitivities

The answers come out to $\partial c/\partial x = 2x$ and $\partial c/\partial y = -2y$. Exactly what we'd get from calculus, but computed mechanically through the graph.

## See It In Motion

Static diagrams are nice, but autodiff really clicks when you *see* the flow. Here's an interactive demo — watch values flow forward, then gradients flow backward:

<div id="autodiff-demo" style="margin: 2em 0; font-family: system-ui, -apple-system, sans-serif;">
  <div id="graph-container" style="width: 100%; max-width: 550px; margin: 0 auto;"></div>

  <div style="text-align: center; margin-top: 1em;">
    <button id="btn-forward" style="padding: 8px 20px; margin: 0 5px; font-size: 14px; cursor: pointer; background: #4CAF50; color: white; border: none; border-radius: 4px;">
      ▶ Forward Pass
    </button>
    <button id="btn-backward" style="padding: 8px 20px; margin: 0 5px; font-size: 14px; cursor: pointer; background: #E91E63; color: white; border: none; border-radius: 4px;">
      ◀ Backward Pass
    </button>
    <button id="btn-reset" style="padding: 8px 20px; margin: 0 5px; font-size: 14px; cursor: pointer; background: #666; color: white; border: none; border-radius: 4px;">
      ↺ Reset
    </button>
  </div>

  <div id="status-text" style="text-align: center; margin-top: 1em; min-height: 3em; color: #666; font-size: 14px; line-height: 1.5;"></div>
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
(function() {
  const width = 550, height = 340;
  const nodeRadius = 24;

  // Theme-aware color schemes (transparent fills, colored strokes)
  const themes = {
    light: {
      input: { fill: 'transparent', stroke: '#5a9fd4' },
      output: { fill: 'transparent', stroke: '#4CAF50' },
      op: { fill: 'transparent', stroke: '#f9a825' },
      text: '#333',
      textMuted: '#666',
      edge: '#bbb',
      arrow: '#999',
      forward: '#4CAF50',
      backward: '#E91E63'
    },
    dark: {
      input: { fill: 'transparent', stroke: '#6ab7ff' },
      output: { fill: 'transparent', stroke: '#66BB6A' },
      op: { fill: 'transparent', stroke: '#ffca28' },
      text: '#e0e0e0',
      textMuted: '#aaa',
      edge: '#666',
      arrow: '#888',
      forward: '#66BB6A',
      backward: '#F48FB1'
    }
  };

  function getTheme() {
    return localStorage.getItem('theme') === 'dark' ? 'dark' : 'light';
  }

  function colors() {
    return themes[getTheme()];
  }

  // Graph data
  const nodes = [
    { id: 'x', label: 'x = 2', x: 100, y: 280, type: 'input', value: 2 },
    { id: 'y', label: 'y = 3', x: 450, y: 280, type: 'input', value: 3 },
    { id: 'sin', label: 'sin', x: 150, y: 170, type: 'op', value: null, gradLabel: '' },
    { id: 'mul', label: '×', x: 320, y: 170, type: 'op', value: null, gradLabel: '' },
    { id: 'add', label: '+', x: 275, y: 60, type: 'output', value: null, gradLabel: '' }
  ];

  const links = [
    { source: 'x', target: 'sin', id: 'x-sin' },
    { source: 'x', target: 'mul', id: 'x-mul' },
    { source: 'y', target: 'mul', id: 'y-mul' },
    { source: 'sin', target: 'add', id: 'sin-add' },
    { source: 'mul', target: 'add', id: 'mul-add' }
  ];

  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));

  // Create SVG
  const svg = d3.select('#graph-container')
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('width', '100%');

  // Arrow marker
  const defs = svg.append('defs');
  defs.append('marker')
    .attr('id', 'arrow')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 8)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-4L10,0L0,4')
    .attr('class', 'arrow-path');

  // Helper to compute edge endpoints (from circle edge to circle edge)
  function linkEndpoints(link) {
    const s = nodeMap[link.source];
    const t = nodeMap[link.target];
    const dx = t.x - s.x, dy = t.y - s.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const ux = dx / dist, uy = dy / dist;
    return {
      x1: s.x + ux * nodeRadius,
      y1: s.y + uy * nodeRadius,
      x2: t.x - ux * (nodeRadius + 6),
      y2: t.y - uy * (nodeRadius + 6)
    };
  }

  // Draw links
  const linkGroup = svg.append('g').attr('class', 'links');
  const linkElements = linkGroup.selectAll('line')
    .data(links)
    .enter()
    .append('line')
    .attr('id', d => `edge-${d.id}`)
    .attr('x1', d => linkEndpoints(d).x1)
    .attr('y1', d => linkEndpoints(d).y1)
    .attr('x2', d => linkEndpoints(d).x2)
    .attr('y2', d => linkEndpoints(d).y2)
    .attr('stroke-width', 2)
    .attr('marker-end', 'url(#arrow)');

  // Draw nodes
  const nodeGroup = svg.append('g').attr('class', 'nodes');
  const nodeElements = nodeGroup.selectAll('g')
    .data(nodes)
    .enter()
    .append('g')
    .attr('transform', d => `translate(${d.x}, ${d.y})`);

  nodeElements.append('circle')
    .attr('r', nodeRadius)
    .attr('class', d => `node-circle node-${d.type}`)
    .attr('stroke-width', 2.5);

  nodeElements.append('text')
    .attr('class', 'node-label')
    .attr('text-anchor', 'middle')
    .attr('dy', d => (d.type === 'op' || d.type === 'output') ? -4 : 5)
    .attr('font-size', 13)
    .attr('font-weight', 'bold')
    .text(d => d.label);

  // Value labels (below node label for ops)
  nodeElements.filter(d => d.type !== 'input')
    .append('text')
    .attr('class', 'value-label')
    .attr('text-anchor', 'middle')
    .attr('dy', 12)
    .attr('font-size', 11);

  // Gradient labels (below nodes for inputs, to the side for ops)
  nodeGroup.selectAll('g')
    .append('text')
    .attr('class', 'grad-label')
    .attr('text-anchor', d => d.type === 'input' ? 'middle' : 'start')
    .attr('x', d => d.type === 'input' ? 0 : (d.id === 'add' ? 40 : 35))
    .attr('y', d => d.type === 'input' ? nodeRadius + 18 : 5)
    .attr('font-size', 11)
    .attr('font-weight', 'bold')
    .style('opacity', 0);

  // Particle group
  const particleGroup = svg.append('g').attr('class', 'particles');

  const statusText = d3.select('#status-text');

  // Apply theme colors
  function applyTheme() {
    const c = colors();

    // Update arrow
    svg.select('.arrow-path').attr('fill', c.arrow);

    // Update edges
    linkElements.attr('stroke', c.edge);

    // Update node circles
    svg.selectAll('.node-input')
      .attr('fill', c.input.fill)
      .attr('stroke', c.input.stroke);
    svg.selectAll('.node-output')
      .attr('fill', c.output.fill)
      .attr('stroke', c.output.stroke);
    svg.selectAll('.node-op')
      .attr('fill', c.op.fill)
      .attr('stroke', c.op.stroke);

    // Update text
    svg.selectAll('.node-label').attr('fill', c.text);
    svg.selectAll('.value-label').attr('fill', c.textMuted);
    svg.selectAll('.grad-label').attr('fill', c.backward);
  }

  function reset() {
    const c = colors();

    // Reset edges
    linkElements.attr('stroke', c.edge).attr('stroke-width', 2);

    // Reset value labels
    nodeGroup.selectAll('.value-label').text('');

    // Reset gradient labels
    nodeGroup.selectAll('.grad-label').style('opacity', 0).text('');

    // Clear particles
    particleGroup.selectAll('*').remove();

    statusText.html(`Click <strong style="color:${c.forward}">Forward Pass</strong> to see values flow up, then <strong style="color:${c.backward}">Backward Pass</strong> to see gradients flow down.`);
  }

  function animateParticle(sourceId, targetId, color, delay, duration) {
    const link = links.find(l => l.source === sourceId && l.target === targetId);
    const ep = linkEndpoints(link);

    setTimeout(() => {
      const particle = particleGroup.append('circle')
        .attr('r', 5)
        .attr('fill', color)
        .attr('cx', ep.x1)
        .attr('cy', ep.y1);

      particle.transition()
        .duration(duration)
        .ease(d3.easeQuadOut)
        .attr('cx', ep.x2)
        .attr('cy', ep.y2)
        .on('end', () => {
          particle.transition().duration(150).style('opacity', 0).remove();
        });
    }, delay);
  }

  function animateParticleReverse(sourceId, targetId, color, delay, duration) {
    const link = links.find(l => l.source === sourceId && l.target === targetId);
    const ep = linkEndpoints(link);

    setTimeout(() => {
      const particle = particleGroup.append('circle')
        .attr('r', 5)
        .attr('fill', color)
        .attr('cx', ep.x2)
        .attr('cy', ep.y2);

      particle.transition()
        .duration(duration)
        .ease(d3.easeQuadOut)
        .attr('cx', ep.x1)
        .attr('cy', ep.y1)
        .on('end', () => {
          particle.transition().duration(150).style('opacity', 0).remove();
        });
    }, delay);
  }

  function highlightEdge(linkId, color, delay) {
    setTimeout(() => {
      d3.select(`#edge-${linkId}`)
        .transition().duration(200)
        .attr('stroke', color)
        .attr('stroke-width', 3);
    }, delay);
  }

  function setValueLabel(nodeId, text, delay) {
    setTimeout(() => {
      nodeGroup.selectAll('g')
        .filter(d => d.id === nodeId)
        .select('.value-label')
        .text(text);
    }, delay);
  }

  function setGradLabel(nodeId, text, delay) {
    setTimeout(() => {
      nodeGroup.selectAll('g')
        .filter(d => d.id === nodeId)
        .select('.grad-label')
        .text(text)
        .transition().duration(200)
        .style('opacity', 1);
    }, delay);
  }

  function forwardPass() {
    reset();
    const c = colors();
    const green = c.forward;
    const speed = 500;

    statusText.html(`<strong style="color:${green}">Forward Pass:</strong> Values flow from inputs → output`);

    setTimeout(() => {
      statusText.html(`<strong style="color:${green}">Step 1:</strong> x flows to sin and ×, y flows to ×`);
    }, 100);

    // Step 1: inputs to ops
    highlightEdge('x-sin', green, 200);
    highlightEdge('x-mul', green, 200);
    highlightEdge('y-mul', green, 200);
    animateParticle('x', 'sin', green, 200, speed);
    animateParticle('x', 'mul', green, 200, speed);
    animateParticle('y', 'mul', green, 200, speed);

    // Step 2: show values
    setTimeout(() => {
      statusText.html(`<strong style="color:${green}">Step 2:</strong> sin(2) ≈ 0.91, 2 × 3 = 6`);
    }, 800);
    setValueLabel('sin', '≈ 0.91', 800);
    setValueLabel('mul', '= 6', 800);

    // Step 3: ops to output
    setTimeout(() => {
      statusText.html(`<strong style="color:${green}">Step 3:</strong> Results flow to addition`);
    }, 1300);
    highlightEdge('sin-add', green, 1400);
    highlightEdge('mul-add', green, 1400);
    animateParticle('sin', 'add', green, 1400, speed);
    animateParticle('mul', 'add', green, 1400, speed);

    setTimeout(() => {
      statusText.html(`<strong style="color:${green}">Done!</strong> Output = 0.91 + 6 ≈ 6.91`);
    }, 2000);
    setValueLabel('add', '≈ 6.91', 2000);
  }

  function backwardPass() {
    reset();
    // Keep forward values visible
    setValueLabel('sin', '≈ 0.91', 0);
    setValueLabel('mul', '= 6', 0);
    setValueLabel('add', '≈ 6.91', 0);

    const c = colors();
    const pink = c.backward;
    const speed = 500;

    statusText.html(`<strong style="color:${pink}">Backward Pass:</strong> Gradients flow from output → inputs`);

    // Step 1: output gradient
    setTimeout(() => {
      statusText.html(`<strong style="color:${pink}">Step 1:</strong> Start at output with gradient = 1`);
    }, 200);
    setGradLabel('add', '∇=1', 200);

    // Step 2: to sin and mul
    setTimeout(() => {
      statusText.html(`<strong style="color:${pink}">Step 2:</strong> + passes gradient 1 to both inputs`);
    }, 900);
    highlightEdge('sin-add', pink, 1000);
    highlightEdge('mul-add', pink, 1000);
    animateParticleReverse('sin', 'add', pink, 1000, speed);
    animateParticleReverse('mul', 'add', pink, 1000, speed);
    setGradLabel('sin', '∇=1', 1600);
    setGradLabel('mul', '∇=1', 1600);

    // Step 3: to x and y
    setTimeout(() => {
      statusText.html(`<strong style="color:${pink}">Step 3:</strong> sin: 1×cos(2)≈−0.42 → x; ×: 1×3→x, 1×2→y`);
    }, 2000);
    highlightEdge('x-sin', pink, 2100);
    highlightEdge('x-mul', pink, 2100);
    highlightEdge('y-mul', pink, 2100);
    animateParticleReverse('x', 'sin', pink, 2100, speed);
    animateParticleReverse('x', 'mul', pink, 2100, speed);
    animateParticleReverse('y', 'mul', pink, 2100, speed);

    setTimeout(() => {
      statusText.html(`<strong style="color:${pink}">Done!</strong> ∂f/∂x = 3 + (−0.42) ≈ 2.58, ∂f/∂y = 2`);
    }, 2700);
    setGradLabel('x', '∇≈2.58', 2700);
    setGradLabel('y', '∇=2', 2700);
  }

  d3.select('#btn-forward').on('click', forwardPass);
  d3.select('#btn-backward').on('click', backwardPass);
  d3.select('#btn-reset').on('click', reset);

  // Initialize with current theme
  applyTheme();
  reset();

  // Listen for theme changes
  window.addEventListener('themechange', function(e) {
    applyTheme();
    reset();
  });
})();
</script>

Try clicking **Forward Pass** first to see how $f(x,y) = x \cdot y + \sin(x)$ computes its value, then **Backward Pass** to watch the chain rule in action — gradients flowing backward, multiplying by local rates, and accumulating where paths merge.

<p style="text-align: center; font-size: 12px; color: #999; margin-top: 0.5em;"><em>Animation crafted with Claude Opus 4.5</em></p>

## The Punchline

Automatic differentiation isn't really about calculus. It's about:

1. **Representation**: Seeing computation as a graph
2. **Locality**: Each operation only knows its own derivative
3. **Composition**: The chain rule falls out naturally from graph traversal
4. **Direction**: Going backward lets us compute all gradients at once

That's the conceptual foundation. In [Part 2]({% post_url 2026-01-17-autodiff-part2-implementation %}), we'll build this in Rust — turning these pictures into actual code.
