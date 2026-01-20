---
layout: post
title:  "Deep Learning from Scratch in Rust, Part 2 — Layers, Models, and Loss"
date:   2026-01-29 10:00:00 -0700
tags: rust machine-learning programming
author: bolu-atx
categories: programming
---

In [Part 1]({% post_url 2026-01-25-deep-learning-scratch-part1-tensors %}), we built tensor autodiff — gradients flow through multi-dimensional arrays with broadcasting and reductions handled correctly. But we still don't have a neural network.

What's missing? The building blocks: **layers** that encapsulate learnable parameters, **models** that compose layers, and **loss functions** that define what "correct" means.

Today we bridge the gap from "autodiff engine" to "trainable model."

<!--more-->

## Variables vs Constants

Not all tensors are equal. Some hold input data (fixed during backward pass), others hold model weights (we need their gradients). The distinction is simple:

```rust
// Variable: tracked for gradients
let weight = Tensor::var("weight", CpuBackend::from_vec(data, shape));

// Constant: not tracked
let input = Tensor::constant(CpuBackend::from_vec(data, shape));
```

`Tensor::var()` creates a named variable node in the computation graph. When we call `backward()`, we get gradients for all variables that influenced the output.

## The Linear Layer

The most fundamental layer: a fully-connected (dense) layer.

$$y = xW^T + b$$

Where:
- $x$: input of shape `[batch, in_features]`
- $W$: weight matrix of shape `[out_features, in_features]`
- $b$: bias vector of shape `[out_features]`
- $y$: output of shape `[batch, out_features]`

```rust
use ad_backend_cpu::CpuBackend;
use ad_tensor::prelude::*;
use rand::Rng;

pub struct Linear {
    /// Weight matrix [out_features, in_features]
    pub weight: Tensor<CpuBackend>,
    /// Bias vector [out_features]
    pub bias: Option<Tensor<CpuBackend>>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let mut rng = rand::thread_rng();

        // Kaiming initialization: std = sqrt(2 / fan_in)
        let std = (2.0 / in_features as f32).sqrt();

        let weight_data: Vec<f32> = (0..out_features * in_features)
            .map(|_| rng.gen::<f32>() * std * 2.0 - std)
            .collect();

        let weight = Tensor::var(
            "weight",
            CpuBackend::from_vec(weight_data, Shape::new(vec![out_features, in_features])),
        );

        let bias = if bias {
            Some(Tensor::var(
                "bias",
                CpuBackend::from_vec(vec![0.0; out_features], Shape::new(vec![out_features])),
            ))
        } else {
            None
        };

        Linear { weight, bias }
    }

    pub fn forward(&self, x: &Tensor<CpuBackend>) -> Tensor<CpuBackend> {
        // x @ W^T
        let y = x.matmul(&self.weight.t());

        // Add bias if present
        match &self.bias {
            Some(bias) => &y + bias,
            None => y,
        }
    }

    pub fn parameters(&self) -> Vec<&Tensor<CpuBackend>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }
}
```

Notice: the layer is tied to `CpuBackend` because initialization uses `rand`. The forward pass itself would work with any backend, but creating random weights requires CPU access. For GPU training, you'd initialize on CPU then transfer.

### Why Kaiming Initialization?

Bad initialization kills training. If weights are too large, activations explode. Too small, gradients vanish.

Kaiming (He) initialization is designed for ReLU networks:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

The $\sqrt{2}$ accounts for ReLU zeroing half the values. This keeps variance stable as signals propagate through layers.

## Activation Functions

Activations introduce non-linearity. Without them, stacking linear layers is pointless — the composition of linear functions is linear.

Activations are pure tensor operations — they work with any backend. We implement them as both tensor methods and standalone functions:

```rust
// In your tensor implementation
impl<B: Backend> Tensor<B> {
    pub fn relu(&self) -> Tensor<B> {
        // max(0, x)
        self.maximum(&Tensor::zeros(self.shape()))
    }

    pub fn sigmoid(&self) -> Tensor<B> {
        // 1 / (1 + exp(-x))
        let one = Tensor::ones(self.shape());
        &one / &(&one + (-self).exp())
    }

    pub fn tanh(&self) -> Tensor<B> {
        // Built-in or: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    }
}
```

Each activation needs a backward implementation in the autodiff engine:

```rust
TensorOp::ReLU => {
    // d/dx ReLU(x) = 1 if x > 0, else 0
    let mask = B::gt(children[0].data(), &B::scalar(0.0));
    vec![Some(B::mul(upstream_grad, &mask))]
}

TensorOp::Sigmoid => {
    // d/dx σ(x) = σ(x)(1 - σ(x))
    let s = output.data();
    let one_minus_s = B::sub(&B::ones(s.shape()), s);
    let local_grad = B::mul(s, &one_minus_s);
    vec![Some(B::mul(upstream_grad, &local_grad))]
}

TensorOp::Tanh => {
    // d/dx tanh(x) = 1 - tanh²(x)
    let t = output.data();
    let t_sq = B::mul(t, t);
    let local_grad = B::sub(&B::ones(t.shape()), &t_sq);
    vec![Some(B::mul(upstream_grad, &local_grad))]
}
```

### Log-Softmax for Classification

Softmax converts logits to probabilities. But we rarely use softmax directly — we use log-softmax for numerical stability:

```rust
pub fn log_softmax<B: Backend>(logits: &Tensor<B>) -> Tensor<B> {
    // log(softmax(x)) = x - log(sum(exp(x)))
    // With max-subtraction for stability:
    let ndim = logits.ndim();
    let axis = if ndim > 0 { ndim - 1 } else { 0 };

    let max_logits = logits.max(Some(&[axis]), true);
    let shifted = logits - &max_logits;
    let log_sum_exp = shifted.exp().sum(Some(&[axis]), true).log();
    shifted - log_sum_exp
}
```

## Building Models

Without a formal `Module` trait, we compose layers manually. This is actually clearer:

```rust
pub struct MLP {
    l1: Linear,
    l2: Linear,
}

impl MLP {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        MLP {
            l1: Linear::new(input_dim, hidden_dim, true),
            l2: Linear::new(hidden_dim, output_dim, true),
        }
    }

    pub fn forward(&self, x: &Tensor<CpuBackend>) -> Tensor<CpuBackend> {
        let h = self.l1.forward(x).relu();
        self.l2.forward(&h)
    }

    pub fn parameters(&self) -> Vec<&Tensor<CpuBackend>> {
        let mut params = self.l1.parameters();
        params.extend(self.l2.parameters());
        params
    }
}
```

No traits, no dynamic dispatch, no `Box<dyn Module>`. Just structs and methods. The Rust compiler can inline everything.

## Loss Functions

Loss functions measure how wrong predictions are. They're the starting point of backpropagation.

Unlike layers (which need random initialization), loss functions are pure tensor operations — they work with any backend.

### Mean Squared Error (MSE)

For regression tasks:

$$\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

```rust
pub fn mse_loss<B: Backend>(pred: &Tensor<B>, target: &Tensor<B>) -> Tensor<B> {
    let diff = pred - target;
    (&diff * &diff).mean(None, false)
}
```

The gradient is straightforward: $\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)$

### Binary Cross-Entropy with Logits

For binary classification, we take raw logits (pre-sigmoid) for numerical stability:

```rust
pub fn binary_cross_entropy_with_logits<B: Backend>(
    logits: &Tensor<B>,
    targets: &Tensor<B>,
) -> Tensor<B> {
    // Numerically stable: max(logits, 0) - logits * targets + log(1 + exp(-|logits|))
    let relu_logits = logits.relu();
    let logits_targets = logits * targets;
    let abs_logits = logits.maximum(&(-logits));
    let one = Tensor::<B>::ones(logits.shape());
    let log_term = (&one + (-&abs_logits).exp()).log();

    let loss = relu_logits - logits_targets + log_term;
    loss.mean(None, false)
}
```

### Soft Cross-Entropy Loss

For multi-class classification with soft labels (probabilities or one-hot):

```rust
pub fn soft_cross_entropy_loss<B: Backend>(
    logits: &Tensor<B>,  // [batch, num_classes]
    targets: &Tensor<B>, // [batch, num_classes] probabilities
) -> Tensor<B> {
    let log_probs = log_softmax(logits);

    // -sum(targets * log_probs) over classes, mean over batch
    let neg_log_probs = -(targets * &log_probs);
    neg_log_probs.sum(Some(&[1]), false).mean(None, false)
}
```

The beautiful gradient: for softmax + cross-entropy, $\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i$ (prediction minus target).

### Visualizing Loss Landscapes

Different losses create different optimization landscapes:

<div id="loss-landscape-viz" style="margin: 2em 0;">
  <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
    <div id="mse-loss-plot" style="width: 320px;"></div>
    <div id="ce-loss-plot" style="width: 320px;"></div>
  </div>
  <div id="loss-status" style="text-align: center; margin-top: 1em; color: #888; font-size: 13px;"></div>
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
(function() {
  const width = 320, height = 280;
  const margin = { top: 30, right: 20, bottom: 40, left: 50 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  function getTheme() {
    return localStorage.getItem('theme') === 'dark' ? 'dark' : 'light';
  }

  const themes = {
    light: { bg: '#fff', text: '#333', grid: '#eee', line: '#e63946', line2: '#2a9d8f' },
    dark: { bg: '#1a1a2e', text: '#e0e0e0', grid: '#333', line: '#ff6b6b', line2: '#4ecdc4' }
  };

  function colors() { return themes[getTheme()]; }

  function mseLoss(p) { return (1 - p) * (1 - p); }
  function mseGrad(p) { return -2 * (1 - p); }
  function ceLoss(p) { return -Math.log(Math.max(p, 1e-7)); }
  function ceGrad(p) { return -1 / Math.max(p, 1e-7); }

  function createPlot(container, title, lossFn, gradFn) {
    const c = colors();
    const svg = d3.select(container)
      .append('svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('width', '100%')
      .style('background', c.bg);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain([0, 1]).range([0, plotW]);
    const yLoss = d3.scaleLinear().domain([0, 4]).range([plotH, 0]);

    const data = d3.range(0.01, 1, 0.01).map(p => ({
      p, loss: lossFn(p), grad: Math.abs(gradFn(p))
    }));

    g.append('g').attr('transform', `translate(0,${plotH})`)
      .call(d3.axisBottom(x).ticks(5)).attr('color', c.text);
    g.append('g').call(d3.axisLeft(yLoss).ticks(5)).attr('color', c.text);

    svg.append('text').attr('x', width / 2).attr('y', 20)
      .attr('text-anchor', 'middle').attr('fill', c.text)
      .attr('font-size', 14).attr('font-weight', 500).text(title);

    g.append('text').attr('x', plotW / 2).attr('y', plotH + 35)
      .attr('text-anchor', 'middle').attr('fill', c.text)
      .attr('font-size', 12).text('Predicted probability (target=1)');

    g.append('text').attr('transform', 'rotate(-90)')
      .attr('x', -plotH / 2).attr('y', -35)
      .attr('text-anchor', 'middle').attr('fill', c.text)
      .attr('font-size', 12).text('Loss');

    const line = d3.line().x(d => x(d.p)).y(d => yLoss(Math.min(d.loss, 4)));
    g.append('path').datum(data).attr('fill', 'none')
      .attr('stroke', c.line).attr('stroke-width', 2.5).attr('d', line);

    const dot = g.append('circle').attr('r', 6).attr('fill', c.line).style('display', 'none');
    const gradLine = g.append('line').attr('stroke', c.line2)
      .attr('stroke-width', 2).attr('stroke-dasharray', '4,2').style('display', 'none');

    svg.on('mousemove', function(event) {
      const [mx] = d3.pointer(event);
      const p = x.invert(mx - margin.left);
      if (p < 0.01 || p > 0.99) { dot.style('display', 'none'); gradLine.style('display', 'none'); return; }

      const loss = lossFn(p), grad = gradFn(p);
      dot.attr('cx', x(p)).attr('cy', yLoss(Math.min(loss, 4))).style('display', 'block');

      const dx = 0.1;
      const x1 = Math.max(0.01, p - dx), x2 = Math.min(0.99, p + dx);
      const y1 = loss + grad * (x1 - p), y2 = loss + grad * (x2 - p);
      gradLine.attr('x1', x(x1)).attr('y1', yLoss(Math.max(0, Math.min(4, y1))))
        .attr('x2', x(x2)).attr('y2', yLoss(Math.max(0, Math.min(4, y2)))).style('display', 'block');

      d3.select('#loss-status').text(`p=${p.toFixed(2)}, loss=${loss.toFixed(3)}, gradient=${grad.toFixed(3)}`);
    });

    svg.on('mouseleave', () => {
      dot.style('display', 'none'); gradLine.style('display', 'none');
      d3.select('#loss-status').text('Hover to see loss and gradient at each prediction');
    });
  }

  createPlot('#mse-loss-plot', 'MSE Loss', mseLoss, mseGrad);
  createPlot('#ce-loss-plot', 'Cross-Entropy Loss', ceLoss, ceGrad);
  d3.select('#loss-status').text('Hover to see loss and gradient at each prediction');
})();
</script>

<p style="text-align: center; font-size: 12px; color: #999; margin-top: 0.5em;"><em>Cross-entropy has steeper gradients for wrong predictions, driving faster learning.</em></p>

Key difference:
- **MSE**: Gradient approaches zero as prediction approaches 0 (confident and wrong). Training stalls.
- **Cross-entropy**: Gradient explodes as prediction approaches 0. Strong signal to correct mistakes.

This is why cross-entropy is preferred for classification.

### When to Use Which Loss

| Task | Output | Activation | Loss |
|------|--------|------------|------|
| Regression | Continuous values | None (linear) | `mse_loss` |
| Binary classification | Probability | Sigmoid | `binary_cross_entropy_with_logits` |
| Multi-class (single label) | Class probabilities | Softmax | `soft_cross_entropy_loss` |

## Putting It Together

A complete training step:

```rust
use ad_tensor::prelude::*;
use ad_backend_cpu::CpuBackend;
use ad_nn::{Linear, mse_loss, Adam};

// Create a simple network
let mut l1 = Linear::new(2, 8, true);
let mut l2 = Linear::new(8, 1, true);
let mut opt = Adam::new(0.01);

// Training data: XOR problem
let inputs = vec![
    vec![0.0, 0.0], vec![0.0, 1.0],
    vec![1.0, 0.0], vec![1.0, 1.0],
];
let targets = vec![0.0, 1.0, 1.0, 0.0];

for epoch in 0..1000 {
    for (input, &target) in inputs.iter().zip(&targets) {
        // Forward pass
        let x = Tensor::var("x", CpuBackend::from_vec(input.clone(), Shape::new(vec![1, 2])));
        let y = Tensor::constant(CpuBackend::from_vec(vec![target], Shape::new(vec![1, 1])));

        let h = l1.forward(&x).relu();
        let pred = l2.forward(&h);
        let loss = mse_loss(&pred, &y);

        // Backward pass
        let grads = loss.backward();

        // Update parameters
        opt.step(&mut l1.weight, grads.wrt(&l1.weight).unwrap());
        if let Some(ref mut bias) = l1.bias {
            opt.step(bias, grads.wrt(bias).unwrap());
        }
        opt.step(&mut l2.weight, grads.wrt(&l2.weight).unwrap());
        if let Some(ref mut bias) = l2.bias {
            opt.step(bias, grads.wrt(bias).unwrap());
        }
    }
}
```

The gradient for every parameter flows automatically through the computation graph — from loss, through the layers, to the weights.

## What's Next

We have models with parameters and loss functions that produce gradients. But gradients alone don't train anything. We need **optimizers** to turn gradients into parameter updates.

[Part 3]({% post_url 2026-02-01-deep-learning-scratch-part3-optimizers %}) implements SGD, Momentum, and Adam — the algorithms that make learning happen.

---

*Part 2 of the "Deep Learning from Scratch in Rust" series. [Part 1]({% post_url 2026-01-25-deep-learning-scratch-part1-tensors %}) covers tensor gradients, [Part 3]({% post_url 2026-02-01-deep-learning-scratch-part3-optimizers %}) covers optimizers.*
