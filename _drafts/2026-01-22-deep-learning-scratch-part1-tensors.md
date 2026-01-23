---
layout: post
title:  "Deep Learning from Scratch in Rust, Part 1 — Tensor Gradients"
date:   2026-01-22 10:00:00 -0700
tags: rust machine-learning programming
author: bolu-atx
categories: programming
---

In the [Autodiff series]({% post_url 2026-01-17-autodiff-rust-part2-scalar-autodiff %}), we built a working autodiff engine for scalar functions. Clean, elegant, and... completely impractical. But building it was so much fun that I decided to take it all the way — from toy scalar engine to a real deep learning framework.

Real neural networks don't operate on individual numbers. They process *tensors* — multi-dimensional arrays where a single forward pass might involve millions of values. Today we'll generalize our scalar engine to tensors and discover the new problems that emerge.

Spoiler: broadcasting is where the elegance gets messy.

<!--more-->

## The Scalar-to-Tensor Leap

Our scalar engine had a simple mental model:

```rust
struct Node {
    value: f64,          // One number
    children: Vec<Expr>,
    op: Op,
}
```

The tensor version looks similar:

```rust
struct TensorNode<B: Backend> {
    data: B::Tensor,     // Many numbers with a shape
    children: Vec<Tensor<B>>,
    op: TensorOp,
}
```

At first glance, we just swap `f64` for a tensor type. All the ops become element-wise. Easy, right?

Not quite. But first — what's with the `B: Backend` generic?

## The Backend Abstraction

Our scalar engine had one implementation. Tensor operations have many: CPU loops, SIMD vectorization, GPU shaders, Apple's Metal, CUDA. We don't want to rewrite the autodiff logic for each.

The solution is a **backend trait** that abstracts tensor operations:

```rust
pub trait Backend: Clone + 'static {
    type Tensor: TensorData;

    // Creation
    fn from_vec(data: Vec<f32>, shape: Shape) -> Self::Tensor;
    fn zeros(shape: &Shape) -> Self::Tensor;
    fn ones(shape: &Shape) -> Self::Tensor;

    // Element-wise ops
    fn add(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn mul(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn exp(a: &Self::Tensor) -> Self::Tensor;

    // Reductions
    fn sum(a: &Self::Tensor, axes: Option<&[usize]>, keepdims: bool) -> Self::Tensor;

    // Linear algebra
    fn matmul(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn transpose(a: &Self::Tensor, axes: Option<&[usize]>) -> Self::Tensor;

    // ... more ops
}
```

A CPU backend implements these with standard loops. A Metal backend dispatches to GPU shaders. The autodiff engine doesn't care — it just calls `B::add()` and the right thing happens.

```rust
pub struct CpuBackend;
pub struct MetalBackend;

impl Backend for CpuBackend {
    type Tensor = CpuTensor;  // Vec<f32> + Shape

    fn add(a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        // Element-wise loop on CPU
    }
}

impl Backend for MetalBackend {
    type Tensor = MetalTensor;  // GPU buffer handle + Shape

    fn add(a: &MetalTensor, b: &MetalTensor) -> MetalTensor {
        // Dispatch Metal compute shader
    }
}
```

This pattern lets us write the autodiff logic once and run it anywhere. The gradient computation, topological sort, chain rule — all backend-agnostic. Only the raw tensor math changes.

Now, back to the new problems that tensors introduce.

## Problem 1: Shape Tracking

Scalars don't have shape. Tensors do, and shapes must be compatible:

```rust
let a = tensor([2, 3]);  // 2×3 matrix
let b = tensor([3, 4]);  // 3×4 matrix

let c = a + b;  // ERROR: shapes don't match
let d = a.matmul(&b);  // OK: [2, 3] @ [3, 4] → [2, 4]
```

Every operation needs shape logic:
- Element-wise: shapes must match (or be broadcastable)
- MatMul: inner dimensions must match
- Reductions: output shape depends on which axes are reduced

The computation graph now carries shape metadata:

```rust
#[derive(Debug, Clone)]
pub struct Shape {
    dims: Vec<usize>,
}

pub trait TensorData {
    fn shape(&self) -> &Shape;
    fn numel(&self) -> usize { self.shape().iter().product() }
}
```

## Problem 2: Broadcasting Semantics

NumPy popularized broadcasting — automatic shape expansion for binary ops:

```python
a = np.array([1, 2, 3])           # [3]
b = np.array([[1], [2], [3]])     # [3, 1]
c = a + b                          # [3, 3]!
```

The rules:
1. Align shapes from the right
2. Each dimension must match or be 1
3. Size-1 dims get "stretched" to match

```
    [3]    pads to  [1, 3]
 [3, 1]             [3, 1]
 ------             ------
 result:            [3, 3]
```

This is convenient for users but creates a gradient problem.

### See Broadcasting in Action

Here's an interactive visualization showing the complete broadcasting flow. Hover over each step to highlight it:

<div id="broadcast-demo" style="margin: 2em 0; font-family: system-ui, -apple-system, sans-serif;">
  <div class="viz-tabs" id="broadcast-tabs">
    <button class="viz-tab active" data-mode="forward">Forward</button>
    <button class="viz-tab" data-mode="backward">Backward</button>
  </div>
  <div id="broadcast-container" style="width: 100%; max-width: 900px; margin: 0 auto;"></div>
  <div id="broadcast-status" style="text-align: center; margin-top: 0.5em; min-height: 1.5em; color: #888; font-size: 13px;"></div>
</div>

<style>
.viz-tabs {
  display: flex;
  justify-content: center;
  gap: 0;
  margin-bottom: 1em;
}
.viz-tab {
  padding: 8px 20px;
  font-size: 13px;
  cursor: pointer;
  background: transparent;
  color: #888;
  border: 1px solid #ddd;
  transition: all 0.15s ease;
}
.viz-tab:first-child {
  border-radius: 4px 0 0 4px;
}
.viz-tab:last-child {
  border-radius: 0 4px 4px 0;
  border-left: none;
}
.viz-tab.active {
  background: #f5f5f5;
  color: #333;
  border-color: #ccc;
}
.viz-tab:hover:not(.active) {
  background: #fafafa;
  color: #555;
}
@media (prefers-color-scheme: dark) {
  .viz-tab {
    color: #777;
    border-color: #444;
  }
  .viz-tab.active {
    background: #333;
    color: #ddd;
    border-color: #555;
  }
  .viz-tab:hover:not(.active) {
    background: #2a2a2a;
    color: #aaa;
  }
}
</style>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
(function() {
  const width = 900, height = 260;
  const cellSize = 32;
  const dimmedOpacity = 0.3;

  const themes = {
    light: {
      text: '#333', textMuted: '#666', cellStroke: '#999',
      tensorA: '#5a9fd4', tensorB: '#f9a825', tensorC: '#4CAF50',
      ghostCell: '#bbb', forward: '#4CAF50', backward: '#E91E63',
      arrow: '#999', stepLabel: '#888'
    },
    dark: {
      text: '#e0e0e0', textMuted: '#aaa', cellStroke: '#666',
      tensorA: '#6ab7ff', tensorB: '#ffca28', tensorC: '#66BB6A',
      ghostCell: '#555', forward: '#66BB6A', backward: '#F48FB1',
      arrow: '#777', stepLabel: '#888'
    }
  };

  function getTheme() {
    return localStorage.getItem('theme') === 'dark' ? 'dark' : 'light';
  }
  function colors() { return themes[getTheme()]; }

  const svg = d3.select('#broadcast-container')
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('width', '100%');

  const statusText = d3.select('#broadcast-status');

  const tensorA = [1, 2, 3];
  const tensorB = [[10], [20], [30]];
  const result = [[11, 12, 13], [21, 22, 23], [31, 32, 33]];

  let mode = 'forward';
  const cols = [60, 320, 620];

  function drawForwardLayout() {
    svg.selectAll('*').remove();
    const c = colors();

    const labels = ['Step 1: Align', 'Step 2: Expand', 'Step 3: Add'];
    const descs = ['Pad a: [3] → [1,3]', 'Stretch size-1 dims', 'Element-wise add'];

    // Create step groups with hover behavior
    const steps = [];
    for (let i = 0; i < 3; i++) {
      const g = svg.append('g')
        .attr('class', 'step-group')
        .attr('data-step', i)
        .style('opacity', 1)
        .style('transition', 'opacity 0.15s ease');
      steps.push(g);
    }

    // Step labels
    labels.forEach((label, i) => {
      steps[i].append('text')
        .attr('x', cols[i] + 100).attr('y', 25)
        .attr('text-anchor', 'middle')
        .attr('font-size', 14).attr('font-weight', 500)
        .attr('fill', c.stepLabel)
        .text(label);
    });

    // Arrows between columns (not part of step groups)
    [0, 1].forEach(i => {
      svg.append('text')
        .attr('x', cols[i] + 230).attr('y', 120)
        .attr('text-anchor', 'middle')
        .attr('font-size', 22).attr('fill', c.arrow)
        .text('→');
    });

    // Step 1: Original tensors aligned
    const step1 = steps[0].append('g').attr('transform', `translate(${cols[0]}, 45)`);
    step1.append('text').attr('x', 50).attr('y', 0)
      .attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', c.tensorA)
      .text('a: [3] → [1,3]');
    const a1 = step1.append('g').attr('transform', `translate(${50 - 1.5*cellSize}, 10)`);
    tensorA.forEach((val, i) => {
      a1.append('rect').attr('x', i*cellSize).attr('y', 0)
        .attr('width', cellSize).attr('height', cellSize)
        .attr('fill', 'transparent').attr('stroke', c.tensorA).attr('stroke-width', 2);
      a1.append('text').attr('x', i*cellSize + cellSize/2).attr('y', cellSize/2 + 5)
        .attr('text-anchor', 'middle').attr('font-size', 14).attr('fill', c.text).text(val);
    });

    step1.append('text').attr('x', 155).attr('y', 0)
      .attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', c.tensorB)
      .text('b: [3,1]');
    const b1 = step1.append('g').attr('transform', `translate(${155 - cellSize/2}, 10)`);
    tensorB.forEach((row, i) => {
      b1.append('rect').attr('x', 0).attr('y', i*cellSize)
        .attr('width', cellSize).attr('height', cellSize)
        .attr('fill', 'transparent').attr('stroke', c.tensorB).attr('stroke-width', 2);
      b1.append('text').attr('x', cellSize/2).attr('y', i*cellSize + cellSize/2 + 5)
        .attr('text-anchor', 'middle').attr('font-size', 14).attr('fill', c.text).text(row[0]);
    });

    // Step 2: Expanded tensors
    const step2 = steps[1].append('g').attr('transform', `translate(${cols[1]}, 45)`);
    step2.append('text').attr('x', 50).attr('y', 0)
      .attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', c.tensorA)
      .text('a → [3,3]');
    const a2 = step2.append('g').attr('transform', `translate(${50 - 1.5*cellSize}, 10)`);
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        const isOrig = row === 0;
        a2.append('rect').attr('x', col*cellSize).attr('y', row*cellSize)
          .attr('width', cellSize).attr('height', cellSize)
          .attr('fill', 'transparent')
          .attr('stroke', isOrig ? c.tensorA : c.ghostCell)
          .attr('stroke-width', isOrig ? 2 : 1)
          .attr('stroke-dasharray', isOrig ? 'none' : '3,2');
        a2.append('text').attr('x', col*cellSize + cellSize/2).attr('y', row*cellSize + cellSize/2 + 5)
          .attr('text-anchor', 'middle').attr('font-size', 13)
          .attr('fill', isOrig ? c.text : c.textMuted).text(tensorA[col]);
      }
    }

    step2.append('text').attr('x', 155).attr('y', 0)
      .attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', c.tensorB)
      .text('b → [3,3]');
    const b2 = step2.append('g').attr('transform', `translate(${155 - 1.5*cellSize}, 10)`);
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        const isOrig = col === 0;
        b2.append('rect').attr('x', col*cellSize).attr('y', row*cellSize)
          .attr('width', cellSize).attr('height', cellSize)
          .attr('fill', 'transparent')
          .attr('stroke', isOrig ? c.tensorB : c.ghostCell)
          .attr('stroke-width', isOrig ? 2 : 1)
          .attr('stroke-dasharray', isOrig ? 'none' : '3,2');
        b2.append('text').attr('x', col*cellSize + cellSize/2).attr('y', row*cellSize + cellSize/2 + 5)
          .attr('text-anchor', 'middle').attr('font-size', 13)
          .attr('fill', isOrig ? c.text : c.textMuted).text(tensorB[row][0]);
      }
    }

    step2.append('text').attr('x', 102).attr('y', 55)
      .attr('text-anchor', 'middle').attr('font-size', 18).attr('fill', c.textMuted)
      .text('+');

    // Step 3: Result
    const step3 = steps[2].append('g').attr('transform', `translate(${cols[2]}, 45)`);
    step3.append('text').attr('x', 100).attr('y', 0)
      .attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', c.tensorC)
      .text('c = a + b: [3,3]');
    const c3 = step3.append('g').attr('transform', `translate(${100 - 1.5*cellSize}, 10)`);
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        c3.append('rect').attr('x', col*cellSize).attr('y', row*cellSize)
          .attr('width', cellSize).attr('height', cellSize)
          .attr('fill', 'transparent').attr('stroke', c.tensorC).attr('stroke-width', 2);
        c3.append('text').attr('x', col*cellSize + cellSize/2).attr('y', row*cellSize + cellSize/2 + 5)
          .attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', c.text)
          .text(result[row][col]);
      }
    }

    // Add invisible hit areas for hover
    steps.forEach((step, idx) => {
      step.append('rect')
        .attr('x', cols[idx]).attr('y', 0)
        .attr('width', 220).attr('height', height)
        .attr('fill', 'transparent')
        .style('cursor', 'pointer');
    });

    // Hover behavior
    steps.forEach((step, idx) => {
      step.on('mouseenter', () => {
        steps.forEach((s, i) => {
          s.style('opacity', i === idx ? 1 : dimmedOpacity);
        });
        statusText.text(descs[idx]);
      });
    });

    svg.on('mouseleave', () => {
      steps.forEach(s => s.style('opacity', 1));
      statusText.text('Hover over each step to highlight');
    });
  }

  function drawBackwardLayout() {
    svg.selectAll('*').remove();
    const c = colors();

    const labels = ['Step 1: Gradient dc', 'Step 2: sum_to(da)', 'Step 3: sum_to(db)'];
    const descs = ['Upstream gradient dc: [3,3]', 'Sum axis 0: da = [3, 3, 3]', 'Sum axis 1: db = [3, 3, 3]ᵀ'];

    const steps = [];
    for (let i = 0; i < 3; i++) {
      const g = svg.append('g')
        .attr('class', 'step-group')
        .attr('data-step', i)
        .style('opacity', 1)
        .style('transition', 'opacity 0.15s ease');
      steps.push(g);
    }

    labels.forEach((label, i) => {
      steps[i].append('text')
        .attr('x', cols[i] + 100).attr('y', 25)
        .attr('text-anchor', 'middle')
        .attr('font-size', 13).attr('font-weight', 500)
        .attr('fill', c.stepLabel)
        .text(label);
    });

    [0, 1].forEach(i => {
      svg.append('text')
        .attr('x', cols[i] + 230).attr('y', 120)
        .attr('text-anchor', 'middle')
        .attr('font-size', 22).attr('fill', c.arrow)
        .text('→');
    });

    // Step 1: dc [3,3]
    const step1 = steps[0].append('g').attr('transform', `translate(${cols[0]}, 45)`);
    step1.append('text').attr('x', 100).attr('y', 0)
      .attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', c.backward)
      .text('dc: [3,3]');
    const dc1 = step1.append('g').attr('transform', `translate(${100 - 1.5*cellSize}, 10)`);
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        dc1.append('rect').attr('x', col*cellSize).attr('y', row*cellSize)
          .attr('width', cellSize).attr('height', cellSize)
          .attr('fill', 'transparent').attr('stroke', c.backward).attr('stroke-width', 2);
        dc1.append('text').attr('x', col*cellSize + cellSize/2).attr('y', row*cellSize + cellSize/2 + 5)
          .attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', c.text).text('1');
      }
    }

    // Step 2: da [3] - result of sum axis 0
    const step2 = steps[1].append('g').attr('transform', `translate(${cols[1]}, 45)`);
    step2.append('text').attr('x', 100).attr('y', 0)
      .attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', c.tensorA)
      .text('da: [3]');
    step2.append('text').attr('x', 100).attr('y', 18)
      .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', c.textMuted)
      .text('Σ over axis 0 (rows)');
    const da2 = step2.append('g').attr('transform', `translate(${100 - 1.5*cellSize}, 30)`);
    [3, 3, 3].forEach((val, i) => {
      da2.append('rect').attr('x', i*cellSize).attr('y', 0)
        .attr('width', cellSize).attr('height', cellSize)
        .attr('fill', 'transparent').attr('stroke', c.tensorA).attr('stroke-width', 2);
      da2.append('text').attr('x', i*cellSize + cellSize/2).attr('y', cellSize/2 + 5)
        .attr('text-anchor', 'middle').attr('font-size', 14).attr('fill', c.text).text(val);
    });

    // Step 3: db [3,1] - result of sum axis 1
    const step3 = steps[2].append('g').attr('transform', `translate(${cols[2]}, 45)`);
    step3.append('text').attr('x', 100).attr('y', 0)
      .attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', c.tensorB)
      .text('db: [3,1]');
    step3.append('text').attr('x', 100).attr('y', 18)
      .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', c.textMuted)
      .text('Σ over axis 1 (cols)');
    const db3 = step3.append('g').attr('transform', `translate(${100 - cellSize/2}, 30)`);
    [3, 3, 3].forEach((val, i) => {
      db3.append('rect').attr('x', 0).attr('y', i*cellSize)
        .attr('width', cellSize).attr('height', cellSize)
        .attr('fill', 'transparent').attr('stroke', c.tensorB).attr('stroke-width', 2);
      db3.append('text').attr('x', cellSize/2).attr('y', i*cellSize + cellSize/2 + 5)
        .attr('text-anchor', 'middle').attr('font-size', 14).attr('fill', c.text).text(val);
    });

    // Add invisible hit areas for hover
    steps.forEach((step, idx) => {
      step.append('rect')
        .attr('x', cols[idx]).attr('y', 0)
        .attr('width', 220).attr('height', height)
        .attr('fill', 'transparent')
        .style('cursor', 'pointer');
    });

    // Hover behavior
    steps.forEach((step, idx) => {
      step.on('mouseenter', () => {
        steps.forEach((s, i) => {
          s.style('opacity', i === idx ? 1 : dimmedOpacity);
        });
        statusText.text(descs[idx]);
      });
    });

    svg.on('mouseleave', () => {
      steps.forEach(s => s.style('opacity', 1));
      statusText.text('Hover over each step to highlight');
    });
  }

  function draw() {
    if (mode === 'forward') {
      drawForwardLayout();
    } else {
      drawBackwardLayout();
    }
    statusText.text('Hover over each step to highlight');
  }

  // Tab handling
  d3.selectAll('#broadcast-tabs .viz-tab').on('click', function() {
    d3.selectAll('#broadcast-tabs .viz-tab').classed('active', false);
    d3.select(this).classed('active', true);
    mode = d3.select(this).attr('data-mode');
    draw();
  });

  draw();
  window.addEventListener('themechange', draw);
})();
</script>

<p style="text-align: center; font-size: 12px; color: #999; margin-top: 0.5em;"><em>Dashed cells show "virtual" copies from broadcasting. Visualization by Claude Opus 4.5</em></p>

### The Broadcast Gradient Problem

Forward pass:
```
a: [3]     broadcasts to [3, 3]
b: [3, 1]  broadcasts to [3, 3]
c = a + b: [3, 3]
```

Backward pass:
```
dc: [3, 3]  ← upstream gradient
da: [???]   ← need shape [3]
db: [???]   ← need shape [3, 1]
```

The gradient `dc` has shape `[3, 3]`, but `a` had shape `[3]`. How do we get back?

The answer: **sum over the broadcast dimensions**.

When `a` was broadcast from `[3]` to `[3, 3]`, each element of `a` contributed to 3 elements of `c`. In the backward pass, we sum those 3 gradient contributions:

```rust
// During forward: a [3] → broadcast → [3, 3]
// During backward: dc [3, 3] → sum axis 0 → da [3]

fn sum_to(grad: &Tensor, target_shape: &Shape) -> Tensor {
    // Find axes that were broadcast (added or stretched)
    // Sum over those axes to reduce back to target_shape
    let mut result = grad.clone();
    for axis in broadcast_axes(grad.shape(), target_shape) {
        result = result.sum(Some(&[axis]), false);
    }
    result.reshape(target_shape)
}
```

This is the key insight: **broadcasting in forward = reduction in backward**.

### Implementing sum_to

```rust
fn compute_broadcast_axes(from: &Shape, to: &Shape) -> Vec<usize> {
    let mut axes = vec![];
    let offset = to.ndim() - from.ndim();

    // Axes that were added (leading dims)
    for i in 0..offset {
        axes.push(i);
    }

    // Axes that were stretched (size 1 → size N)
    for i in 0..from.ndim() {
        if from.dim(i) == 1 && to.dim(i + offset) > 1 {
            axes.push(i + offset);
        }
    }

    axes
}

fn sum_to<B: Backend>(grad: &B::Tensor, target: &Shape) -> B::Tensor {
    let axes = compute_broadcast_axes(target, grad.shape());
    if axes.is_empty() {
        return B::clone_tensor(grad);
    }

    let mut result = B::clone_tensor(grad);
    // Sum in reverse order to keep axis indices valid
    for &axis in axes.iter().rev() {
        result = B::sum(&result, Some(&[axis]), false);
    }

    // Reshape in case we summed to scalar-like shape
    B::reshape(&result, target)
}
```

Every binary operation that broadcasts must use `sum_to` in its backward pass:

```rust
TensorOp::Add => {
    // Forward: broadcast both to output shape, add
    // Backward: sum gradients back to input shapes
    vec![
        Some(B::sum_to(upstream_grad, children[0].shape())),
        Some(B::sum_to(upstream_grad, children[1].shape())),
    ]
}

TensorOp::Mul => {
    // d(a*b)/da = b, d(a*b)/db = a
    // But we need to broadcast, multiply, then sum_to
    let b_expanded = B::broadcast_to(children[1].data(), output_shape);
    let a_expanded = B::broadcast_to(children[0].data(), output_shape);

    let local_a = B::mul(upstream_grad, &b_expanded);
    let local_b = B::mul(upstream_grad, &a_expanded);

    vec![
        Some(B::sum_to(&local_a, children[0].shape())),
        Some(B::sum_to(&local_b, children[1].shape())),
    ]
}
```

## Problem 3: Reduction Operations

Reductions (sum, mean, max) collapse dimensions:

```rust
let x = tensor([[1, 2, 3],
                [4, 5, 6]]);  // [2, 3]

x.sum(None, false)         // → scalar (sum all)
x.sum(Some(&[0]), false)   // → [3] (sum rows)
x.sum(Some(&[1]), false)   // → [2] (sum cols)
x.sum(Some(&[1]), true)    // → [2, 1] (keepdims)
```

### Reduction and Gradient Flow

Reductions collapse dimensions going forward. The gradient does the reverse — it **expands** back to the original shape:

<div id="reduction-demo" style="margin: 2em 0; font-family: system-ui, -apple-system, sans-serif;">
  <div class="viz-tabs" id="reduction-tabs">
    <button class="viz-tab active" data-mode="forward">Forward</button>
    <button class="viz-tab" data-mode="backward">Backward</button>
  </div>
  <div id="reduction-container" style="width: 100%; max-width: 900px; margin: 0 auto;"></div>
  <div id="reduction-status" style="text-align: center; margin-top: 0.5em; min-height: 1.5em; color: #888; font-size: 13px;"></div>
</div>

<script>
(function() {
  const width = 900, height = 220;
  const cellSize = 34;
  const dimmedOpacity = 0.3;

  const themes = {
    light: {
      text: '#333', textMuted: '#666', cellStroke: '#999',
      tensorX: '#5a9fd4', tensorY: '#4CAF50', grad: '#E91E63',
      ghostCell: '#bbb', forward: '#4CAF50', backward: '#E91E63',
      arrow: '#999', stepLabel: '#888'
    },
    dark: {
      text: '#e0e0e0', textMuted: '#aaa', cellStroke: '#666',
      tensorX: '#6ab7ff', tensorY: '#66BB6A', grad: '#F48FB1',
      ghostCell: '#555', forward: '#66BB6A', backward: '#F48FB1',
      arrow: '#777', stepLabel: '#888'
    }
  };

  function getTheme() {
    return localStorage.getItem('theme') === 'dark' ? 'dark' : 'light';
  }
  function colors() { return themes[getTheme()]; }

  const svg = d3.select('#reduction-container')
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('width', '100%');

  const statusText = d3.select('#reduction-status');

  const tensorX = [[1, 2, 3], [4, 5, 6]];
  const tensorY = [5, 7, 9];
  const colColors = ['#e57373', '#81c784', '#64b5f6'];

  let mode = 'forward';
  const cols = [50, 330, 620];

  function drawForwardLayout() {
    svg.selectAll('*').remove();
    const c = colors();

    const labels = ['Step 1: Input', 'Step 2: Sum axis=0', 'Step 3: Result'];
    const descs = ['Input tensor x: [2, 3]', 'Sum down columns: 1+4, 2+5, 3+6', 'Result y: [3] — rows collapsed'];

    const steps = [];
    for (let i = 0; i < 3; i++) {
      const g = svg.append('g')
        .attr('class', 'step-group')
        .attr('data-step', i)
        .style('opacity', 1)
        .style('transition', 'opacity 0.15s ease');
      steps.push(g);
    }

    labels.forEach((label, i) => {
      steps[i].append('text')
        .attr('x', cols[i] + 90).attr('y', 25)
        .attr('text-anchor', 'middle')
        .attr('font-size', 14).attr('font-weight', 500)
        .attr('fill', c.stepLabel)
        .text(label);
    });

    // Arrows
    [0, 1].forEach(i => {
      svg.append('text').attr('x', cols[i] + 220).attr('y', 100)
        .attr('text-anchor', 'middle').attr('font-size', 22).attr('fill', c.arrow)
        .text('→');
    });

    // Step 1: Input tensor x [2,3]
    const step1 = steps[0].append('g').attr('transform', `translate(${cols[0]}, 45)`);
    step1.append('text').attr('x', 90).attr('y', 0)
      .attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', c.tensorX)
      .text('x: [2, 3]');

    const xGroup = step1.append('g').attr('transform', `translate(${90 - 1.5*cellSize}, 10)`);
    for (let row = 0; row < 2; row++) {
      for (let col = 0; col < 3; col++) {
        xGroup.append('rect').attr('x', col*cellSize).attr('y', row*cellSize)
          .attr('width', cellSize).attr('height', cellSize)
          .attr('fill', 'transparent').attr('stroke', colColors[col]).attr('stroke-width', 2);
        xGroup.append('text').attr('x', col*cellSize + cellSize/2).attr('y', row*cellSize + cellSize/2 + 5)
          .attr('text-anchor', 'middle').attr('font-size', 15).attr('fill', c.text)
          .text(tensorX[row][col]);
      }
    }

    // Step 2: Show summing operation
    const step2 = steps[1].append('g').attr('transform', `translate(${cols[1]}, 45)`);
    step2.append('text').attr('x', 90).attr('y', 0)
      .attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', c.textMuted)
      .text('sum(axis=0)');

    const sumGroup = step2.append('g').attr('transform', `translate(${90 - 1.5*cellSize}, 10)`);
    for (let row = 0; row < 2; row++) {
      for (let col = 0; col < 3; col++) {
        sumGroup.append('rect').attr('x', col*cellSize).attr('y', row*cellSize)
          .attr('width', cellSize).attr('height', cellSize)
          .attr('fill', 'transparent').attr('stroke', colColors[col]).attr('stroke-width', 1)
          .attr('stroke-dasharray', '3,2');
        sumGroup.append('text').attr('x', col*cellSize + cellSize/2).attr('y', row*cellSize + cellSize/2 + 5)
          .attr('text-anchor', 'middle').attr('font-size', 14).attr('fill', c.textMuted)
          .text(tensorX[row][col]);
      }
    }
    // Sum arrows
    for (let col = 0; col < 3; col++) {
      sumGroup.append('text').attr('x', col*cellSize + cellSize/2).attr('y', 2*cellSize + 18)
        .attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', colColors[col])
        .text('↓');
    }

    // Step 3: Result y [3]
    const step3 = steps[2].append('g').attr('transform', `translate(${cols[2]}, 45)`);
    step3.append('text').attr('x', 90).attr('y', 0)
      .attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', c.tensorY)
      .text('y: [3]');

    const yGroup = step3.append('g').attr('transform', `translate(${90 - 1.5*cellSize}, 25)`);
    tensorY.forEach((val, i) => {
      yGroup.append('rect').attr('x', i*cellSize).attr('y', 0)
        .attr('width', cellSize).attr('height', cellSize)
        .attr('fill', 'transparent').attr('stroke', colColors[i]).attr('stroke-width', 2);
      yGroup.append('text').attr('x', i*cellSize + cellSize/2).attr('y', cellSize/2 + 5)
        .attr('text-anchor', 'middle').attr('font-size', 15).attr('fill', c.text)
        .text(val);
    });

    // Add invisible hit areas for hover
    steps.forEach((step, idx) => {
      step.append('rect')
        .attr('x', cols[idx]).attr('y', 0)
        .attr('width', 220).attr('height', height)
        .attr('fill', 'transparent')
        .style('cursor', 'pointer');
    });

    // Hover behavior
    steps.forEach((step, idx) => {
      step.on('mouseenter', () => {
        steps.forEach((s, i) => {
          s.style('opacity', i === idx ? 1 : dimmedOpacity);
        });
        statusText.text(descs[idx]);
      });
    });

    svg.on('mouseleave', () => {
      steps.forEach(s => s.style('opacity', 1));
      statusText.text('Hover over each step to highlight');
    });
  }

  function drawBackwardLayout() {
    svg.selectAll('*').remove();
    const c = colors();

    const labels = ['Step 1: Gradient dy', 'Step 2: Broadcast', 'Step 3: Gradient dx'];
    const descs = ['Upstream gradient dy: [3]', 'Broadcast copies to each row', 'Gradient dx: [2, 3] — same per column'];

    const steps = [];
    for (let i = 0; i < 3; i++) {
      const g = svg.append('g')
        .attr('class', 'step-group')
        .attr('data-step', i)
        .style('opacity', 1)
        .style('transition', 'opacity 0.15s ease');
      steps.push(g);
    }

    labels.forEach((label, i) => {
      steps[i].append('text')
        .attr('x', cols[i] + 90).attr('y', 25)
        .attr('text-anchor', 'middle')
        .attr('font-size', 14).attr('font-weight', 500)
        .attr('fill', c.stepLabel)
        .text(label);
    });

    // Arrows
    [0, 1].forEach(i => {
      svg.append('text').attr('x', cols[i] + 220).attr('y', 100)
        .attr('text-anchor', 'middle').attr('font-size', 22).attr('fill', c.arrow)
        .text('→');
    });

    const gradVals = ['g₁', 'g₂', 'g₃'];

    // Step 1: dy [3]
    const step1 = steps[0].append('g').attr('transform', `translate(${cols[0]}, 45)`);
    step1.append('text').attr('x', 90).attr('y', 0)
      .attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', c.backward)
      .text('dy: [3]');

    const dyGroup = step1.append('g').attr('transform', `translate(${90 - 1.5*cellSize}, 25)`);
    gradVals.forEach((val, i) => {
      dyGroup.append('rect').attr('x', i*cellSize).attr('y', 0)
        .attr('width', cellSize).attr('height', cellSize)
        .attr('fill', 'transparent').attr('stroke', c.backward).attr('stroke-width', 2);
      dyGroup.append('text').attr('x', i*cellSize + cellSize/2).attr('y', cellSize/2 + 5)
        .attr('text-anchor', 'middle').attr('font-size', 14).attr('fill', c.text)
        .text(val);
    });

    // Step 2: Show broadcast operation
    const step2 = steps[1].append('g').attr('transform', `translate(${cols[1]}, 45)`);
    step2.append('text').attr('x', 90).attr('y', 0)
      .attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', c.textMuted)
      .text('broadcast');

    const bcGroup = step2.append('g').attr('transform', `translate(${90 - 1.5*cellSize}, 25)`);
    gradVals.forEach((val, i) => {
      bcGroup.append('rect').attr('x', i*cellSize).attr('y', 0)
        .attr('width', cellSize).attr('height', cellSize)
        .attr('fill', 'transparent').attr('stroke', c.backward).attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,2');
      bcGroup.append('text').attr('x', i*cellSize + cellSize/2).attr('y', cellSize/2 + 5)
        .attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', c.textMuted)
        .text(val);
      // Broadcast arrows
      bcGroup.append('text').attr('x', i*cellSize + cellSize/2).attr('y', cellSize + 18)
        .attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', c.backward)
        .text('↓');
    });

    // Step 3: dx [2,3]
    const step3 = steps[2].append('g').attr('transform', `translate(${cols[2]}, 45)`);
    step3.append('text').attr('x', 90).attr('y', 0)
      .attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', c.tensorX)
      .text('dx: [2, 3]');

    const dxGroup = step3.append('g').attr('transform', `translate(${90 - 1.5*cellSize}, 10)`);
    for (let row = 0; row < 2; row++) {
      for (let col = 0; col < 3; col++) {
        const isOrig = row === 0;
        dxGroup.append('rect').attr('x', col*cellSize).attr('y', row*cellSize)
          .attr('width', cellSize).attr('height', cellSize)
          .attr('fill', 'transparent')
          .attr('stroke', isOrig ? c.tensorX : c.ghostCell)
          .attr('stroke-width', isOrig ? 2 : 1)
          .attr('stroke-dasharray', isOrig ? 'none' : '3,2');
        dxGroup.append('text').attr('x', col*cellSize + cellSize/2).attr('y', row*cellSize + cellSize/2 + 5)
          .attr('text-anchor', 'middle').attr('font-size', 14)
          .attr('fill', isOrig ? c.text : c.textMuted)
          .text(gradVals[col]);
      }
    }

    // Add invisible hit areas for hover
    steps.forEach((step, idx) => {
      step.append('rect')
        .attr('x', cols[idx]).attr('y', 0)
        .attr('width', 220).attr('height', height)
        .attr('fill', 'transparent')
        .style('cursor', 'pointer');
    });

    // Hover behavior
    steps.forEach((step, idx) => {
      step.on('mouseenter', () => {
        steps.forEach((s, i) => {
          s.style('opacity', i === idx ? 1 : dimmedOpacity);
        });
        statusText.text(descs[idx]);
      });
    });

    svg.on('mouseleave', () => {
      steps.forEach(s => s.style('opacity', 1));
      statusText.text('Hover over each step to highlight');
    });
  }

  function draw() {
    if (mode === 'forward') {
      drawForwardLayout();
    } else {
      drawBackwardLayout();
    }
    statusText.text('Hover over each step to highlight');
  }

  // Tab handling
  d3.selectAll('#reduction-tabs .viz-tab').on('click', function() {
    d3.selectAll('#reduction-tabs .viz-tab').classed('active', false);
    d3.select(this).classed('active', true);
    mode = d3.select(this).attr('data-mode');
    draw();
  });

  draw();
  window.addEventListener('themechange', draw);
})();
</script>

<p style="text-align: center; font-size: 12px; color: #999; margin-top: 0.5em;"><em>Reduction collapses axes; gradient broadcast restores them. Visualization by Claude Opus 4.5</em></p>

The gradient for sum is straightforward — broadcast back to input shape:

```rust
TensorOp::Sum { axes, keepdims } => {
    let input_shape = children[0].shape();

    let expanded = if *keepdims {
        upstream_grad.clone()
    } else {
        // Need to unsqueeze removed axes first
        expand_for_broadcast(upstream_grad, input_shape, axes)
    };

    vec![Some(B::broadcast_to(&expanded, input_shape))]
}
```

Mean divides by the count:

```rust
TensorOp::Mean { axes, keepdims } => {
    let count: usize = match axes {
        Some(ax) => ax.iter().map(|&i| input_shape.dim(i)).product(),
        None => input_shape.numel(),
    };

    let sum_grad = /* same as Sum */;
    vec![Some(B::div(&sum_grad, &B::scalar(count as f32)))]
}
```

Max/Min are trickier — gradient only flows to the maximum element(s):

```rust
TensorOp::Max { axes, keepdims } => {
    // Create mask: 1 where input == max value, 0 elsewhere
    let expanded_max = expand_and_broadcast(output, input_shape, axes);
    let mask = B::eq(children[0].data(), &expanded_max);

    let expanded_grad = expand_and_broadcast(upstream_grad, input_shape, axes);
    vec![Some(B::mul(&expanded_grad, &mask))]
}
```

## Problem 4: Matrix Multiplication Gradients

MatMul is the workhorse of deep learning. For `C = A @ B`:

```
A: [M, K]
B: [K, N]
C: [M, N]
```

The gradients involve transposes:

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot B^T$$

$$\frac{\partial L}{\partial B} = A^T \cdot \frac{\partial L}{\partial C}$$

```rust
TensorOp::MatMul => {
    let a = children[0].data();
    let b = children[1].data();

    // dC is [M, N]
    // dA = dC @ B^T → [M, N] @ [N, K] → [M, K] ✓
    // dB = A^T @ dC → [K, M] @ [M, N] → [K, N] ✓
    let grad_a = B::matmul(upstream_grad, &B::transpose(b, None));
    let grad_b = B::matmul(&B::transpose(a, None), upstream_grad);

    vec![Some(grad_a), Some(grad_b)]
}
```

Batched matmul (e.g., `[batch, M, K] @ [batch, K, N]`) follows the same pattern but operates on the last two dimensions.

## The Complete Backward Pass

With these pieces, the backward algorithm is the same as scalars:

```rust
pub fn backward<B: Backend>(output: &Tensor<B>) -> Gradients<B> {
    let topo_order = topological_sort(output);

    // Key difference: initialize with ones(shape) not 1.0
    let mut adjoints: HashMap<NodeId, B::Tensor> = HashMap::new();
    adjoints.insert(output.id(), B::ones(output.shape()));

    for expr in topo_order.iter().rev() {
        let Some(upstream) = adjoints.get(&expr.id()) else { continue };
        let upstream = B::clone_tensor(upstream);

        let child_grads = compute_local_gradients::<B>(expr, &upstream);

        for (child, grad) in expr.children().iter().zip(child_grads) {
            if let Some(grad) = grad {
                adjoints
                    .entry(child.id())
                    .and_modify(|acc| B::accumulate_grad(acc, &grad))
                    .or_insert(grad);
            }
        }
    }

    // ...
}
```

The differences from scalar autodiff:
1. `B::ones(shape)` instead of `1.0`
2. `compute_local_gradients` returns tensors, handles broadcasting
3. `accumulate_grad` adds tensors element-wise

## Testing Tensor Gradients

Numerical gradient checking still works, just with more points:

```rust
fn numerical_gradient<B, F>(f: F, x: &Tensor<B>, eps: f32) -> B::Tensor
where
    F: Fn(&Tensor<B>) -> Tensor<B>,
{
    let mut grad_data = vec![0.0; x.numel()];

    for i in 0..x.numel() {
        // Perturb element i by ±eps
        let mut plus_data = x.as_slice().to_vec();
        let mut minus_data = x.as_slice().to_vec();
        plus_data[i] += eps;
        minus_data[i] -= eps;

        let plus = Tensor::from_vec(plus_data, x.shape().clone());
        let minus = Tensor::from_vec(minus_data, x.shape().clone());

        let plus_out = f(&plus).sum(None, false).item();
        let minus_out = f(&minus).sum(None, false).item();

        grad_data[i] = (plus_out - minus_out) / (2.0 * eps);
    }

    B::from_vec(grad_data, x.shape().clone())
}
```

This is O(n) forward passes for n elements — slow, but good for testing.

## Summary: What's New with Tensors

| Scalar Autodiff | Tensor Autodiff |
|-----------------|-----------------|
| Values are `f64` | Values are `Tensor` with shape |
| Ops take scalars | Ops handle broadcasting |
| Gradients are scalars | Gradients are tensors, same shape as inputs |
| — | Need `sum_to` for broadcast gradients |
| — | Need `expand` for reduction gradients |
| — | MatMul gradients involve transposes |

The core algorithm (topo sort, reverse traversal, chain rule) is unchanged. The bookkeeping around shapes is what makes tensor autodiff more complex.

## What's Next

We have tensors with gradients. But an autodiff engine isn't a neural network yet. We need:
- **Layers**: Encapsulate parameters and forward computation
- **Models**: Compose layers into trainable architectures
- **Loss functions**: Define what "correct" means

[Part 2]({% post_url 2026-01-29-deep-learning-scratch-part2-models %}) builds these building blocks.

---

*Part 1 of the "Deep Learning from Scratch in Rust" series. This builds on the [Autodiff in Rust]({% post_url 2026-01-17-autodiff-rust-part2-scalar-autodiff %}) series. [Part 2]({% post_url 2026-01-29-deep-learning-scratch-part2-models %}) covers models and loss.*
