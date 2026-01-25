---
layout: post
title:  "Deep Learning from Scratch in Rust, Part 3 — Optimizers"
date:   2025-12-15 10:00:00 -0700
tags: rust machine-learning programming
author: bolu-atx
categories: programming
---

We have gradients. Now what?

In [Part 2]({% post_url 2025-11-15-deep-learning-scratch-part2-models %}), we built layers, models, and loss functions. Given a model and a loss, autodiff computes ∂loss/∂θ for every parameter θ. But gradients alone don't train a model. We need an *optimizer* to turn gradients into parameter updates.

Today we'll implement the three most important optimizers: SGD, SGD with Momentum, and Adam. Along the way, we'll see why Adam became the default choice.

<!--more-->

## The Optimization Problem

Training a neural network means minimizing a loss function:

$$\theta^* = \arg\min_\theta \mathcal{L}(\theta)$$

Where θ represents all trainable parameters (weights, biases). The gradient ∇L(θ) points in the direction of steepest *increase*, so we move in the opposite direction:

$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$$

This is gradient descent. The learning rate η controls step size.

## SGD: The Simplest Optimizer

Stochastic Gradient Descent is the baseline:

$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$

Where $g_t = \nabla \mathcal{L}(\theta_t)$ is the gradient.

```rust
use ad_backend_cpu::CpuBackend;
use ad_tensor::prelude::*;
use std::collections::HashMap;

pub struct SGD {
    pub lr: f32,
    pub momentum: f32,
    velocities: HashMap<NodeId, Vec<f32>>,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        SGD { lr, momentum: 0.0, velocities: HashMap::new() }
    }

    pub fn with_momentum(lr: f32, momentum: f32) -> Self {
        SGD { lr, momentum, velocities: HashMap::new() }
    }

    pub fn step(&mut self, param: &mut Tensor<CpuBackend>, grad: &<CpuBackend as Backend>::Tensor) {
        let param_id = param.id();
        let param_data = param.data().as_slice();
        let grad_data = grad.as_slice();
        let mut new_data = param_data.to_vec();

        if self.momentum > 0.0 {
            let velocity = self.velocities
                .entry(param_id)
                .or_insert_with(|| vec![0.0; param_data.len()]);

            for i in 0..new_data.len() {
                velocity[i] = self.momentum * velocity[i] + grad_data[i];
                new_data[i] -= self.lr * velocity[i];
            }
        } else {
            for i in 0..new_data.len() {
                new_data[i] -= self.lr * grad_data[i];
            }
        }

        *param = Tensor::var(
            param.var_name().unwrap_or("param"),
            CpuBackend::from_vec(new_data, param.shape().clone()),
        );
    }
}
```

Notice: no traits, no generics. The optimizer mutates the parameter tensor directly. Internal state (velocity) uses plain `Vec<f32>` — no need for backend tensors here.

SGD is simple but has problems:
- Gets stuck in flat regions (gradient ≈ 0)
- Oscillates in narrow valleys
- Learning rate is sensitive — too high diverges, too low is slow

### Visualizing the Problem

Consider a narrow valley — the classic pathological case for optimization:

<svg viewBox="0 0 500 320" style="width: 100%; max-width: 500px; display: block; margin: 2rem auto; font-family: system-ui, sans-serif;">
  <!-- Background -->
  <rect width="500" height="320" fill="var(--bg-secondary, #f5f5f5)"/>

  <!-- Contour lines (elliptical valley) -->
  <g stroke="var(--text-tertiary, #ccc)" fill="none" stroke-width="1">
    <ellipse cx="400" cy="160" rx="30" ry="15" />
    <ellipse cx="400" cy="160" rx="60" ry="30" />
    <ellipse cx="400" cy="160" rx="100" ry="50" />
    <ellipse cx="400" cy="160" rx="150" ry="75" />
    <ellipse cx="400" cy="160" rx="210" ry="105" />
    <ellipse cx="400" cy="160" rx="280" ry="140" />
  </g>

  <!-- Minimum marker -->
  <circle cx="400" cy="160" r="6" fill="none" stroke="var(--text-primary, #333)" stroke-width="2"/>
  <text x="400" y="145" text-anchor="middle" font-size="11" fill="var(--text-secondary, #666)">minimum</text>

  <!-- SGD path (red) - zig-zag oscillation -->
  <path d="M 60,60 L 90,110 L 110,70 L 140,120 L 160,80 L 190,130 L 210,90 L 240,140 L 260,100 L 290,145 L 310,115 L 340,150 L 360,130 L 380,155"
        fill="none" stroke="#e63946" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="380" cy="155" r="5" fill="#e63946"/>
  <circle cx="60" cy="60" r="4" fill="#e63946" opacity="0.5"/>

  <!-- Momentum path (green) - smoother with overshoot -->
  <path d="M 60,60 L 100,100 L 150,130 L 220,155 L 300,165 L 380,170 L 420,162 L 405,160"
        fill="none" stroke="#2a9d8f" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="405" cy="160" r="5" fill="#2a9d8f"/>
  <circle cx="60" cy="60" r="4" fill="#2a9d8f" opacity="0.5"/>

  <!-- Adam path (yellow) - efficient direct path -->
  <path d="M 60,60 L 120,100 L 200,135 L 300,155 L 400,160"
        fill="none" stroke="#d4a017" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="400" cy="160" r="5" fill="#d4a017"/>
  <circle cx="60" cy="60" r="4" fill="#d4a017" opacity="0.5"/>

  <!-- Start label -->
  <text x="60" y="50" text-anchor="middle" font-size="11" fill="var(--text-secondary, #666)">start</text>

  <!-- Legend -->
  <g transform="translate(60, 280)">
    <circle cx="0" cy="0" r="5" fill="#e63946"/>
    <text x="12" y="4" font-size="12" fill="var(--text-primary, #333)">SGD — oscillates across valley</text>
  </g>
  <g transform="translate(60, 300)">
    <circle cx="0" cy="0" r="5" fill="#2a9d8f"/>
    <text x="12" y="4" font-size="12" fill="var(--text-primary, #333)">Momentum — overshoots, then corrects</text>
  </g>
  <g transform="translate(300, 280)">
    <circle cx="0" cy="0" r="5" fill="#d4a017"/>
    <text x="12" y="4" font-size="12" fill="var(--text-primary, #333)">Adam — adaptive, efficient</text>
  </g>
</svg>

SGD oscillates back and forth across the narrow valley — the gradient points perpendicular to the valley walls, not toward the minimum. Momentum builds velocity in the consistent direction while dampening oscillations. Adam adapts per-parameter, taking larger steps where gradients are stable.

## SGD with Momentum

Momentum adds "velocity" to the parameter updates:

$$v_{t+1} = \mu \cdot v_t + g_t$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_{t+1}$$

The velocity accumulates gradients over time. If gradients consistently point the same direction, velocity builds up. If they oscillate, they cancel out.

We already showed this above — SGD with momentum is just `SGD::with_momentum(lr, 0.9)`:

```rust
let mut opt = SGD::with_momentum(0.01, 0.9);

// In the step function:
// v = momentum * v + grad
// param = param - lr * v
```

Typical momentum value: 0.9 (90% of previous velocity retained).

## Adam: The Modern Default

Adam (Adaptive Moment Estimation) combines momentum with per-parameter adaptive learning rates.

The key insight: some parameters need larger updates than others. Adam tracks:
- First moment (mean of gradients) — like momentum
- Second moment (variance of gradients) — for adaptive scaling

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

Then bias-correct (important for early iterations):

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Finally, update with adaptive step:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Parameters with high variance (unstable gradients) get smaller updates. Parameters with consistent gradients get larger updates.

```rust
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    m: HashMap<NodeId, Vec<f32>>,  // First moment
    v: HashMap<NodeId, Vec<f32>>,  // Second moment
    t: u64,                         // Step counter
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    pub fn step(&mut self, param: &mut Tensor<CpuBackend>, grad: &<CpuBackend as Backend>::Tensor) {
        self.t += 1;

        let param_id = param.id();
        let param_data = param.data().as_slice();
        let grad_data = grad.as_slice();
        let n = param_data.len();

        // Initialize moment estimates if needed
        let m = self.m.entry(param_id).or_insert_with(|| vec![0.0; n]);
        let v = self.v.entry(param_id).or_insert_with(|| vec![0.0; n]);

        let mut new_data = param_data.to_vec();

        // Bias correction factors
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..n {
            // Update biased first moment: m = β₁ * m + (1 - β₁) * g
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad_data[i];

            // Update biased second moment: v = β₂ * v + (1 - β₂) * g²
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad_data[i] * grad_data[i];

            // Bias-corrected estimates
            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;

            // Update: θ = θ - lr * m_hat / (sqrt(v_hat) + ε)
            new_data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }

        *param = Tensor::var(
            param.var_name().unwrap_or("param"),
            CpuBackend::from_vec(new_data, param.shape().clone()),
        );
    }
}
```

Like SGD, Adam stores its internal state as plain `Vec<f32>`. No need for backend tensors — the optimizer logic is CPU-only.

### Why Adam Works

1. **Momentum from m**: Smooths out gradient noise
2. **Adaptivity from v**: Learns different rates per parameter
3. **Bias correction**: Prevents early updates from being too small

Default hyperparameters (β₁=0.9, β₂=0.999, ε=1e-8) work well for most problems.

## AdamW: Weight Decay Done Right

Original Adam has a subtle bug with L2 regularization. Weight decay should shrink parameters toward zero:

$$\theta_{t+1} = (1 - \lambda) \cdot \theta_t - \eta \cdot \text{update}$$

But if you add L2 to the loss, Adam's adaptive scaling weakens the regularization effect. AdamW fixes this by decoupling weight decay — applying it directly to parameters rather than through the gradient:

```rust
// In AdamW's step function, before the normal Adam update:
new_data[i] *= 1.0 - self.lr * self.weight_decay;
// Then apply normal Adam update
new_data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
```

## The Training Loop

Putting it all together:

```rust
use ad_tensor::prelude::*;
use ad_backend_cpu::CpuBackend;
use ad_nn::{Linear, mse_loss, Adam};

// Create layers
let mut l1 = Linear::new(2, 8, true);
let mut l2 = Linear::new(8, 1, true);
let mut opt = Adam::new(0.01);

// Training loop
for epoch in 0..1000 {
    for (input_data, target_data) in &dataset {
        // 1. Create tensors
        let x = Tensor::var("x", CpuBackend::from_vec(input_data.clone(), Shape::new(vec![1, 2])));
        let y = Tensor::constant(CpuBackend::from_vec(vec![*target_data], Shape::new(vec![1, 1])));

        // 2. Forward pass
        let h = l1.forward(&x).relu();
        let pred = l2.forward(&h);

        // 3. Compute loss
        let loss = mse_loss(&pred, &y);

        // 4. Backward pass
        let grads = loss.backward();

        // 5. Update each parameter
        opt.step(&mut l1.weight, grads.wrt(&l1.weight).unwrap());
        if let Some(ref mut bias) = l1.bias {
            opt.step(bias, grads.wrt(bias).unwrap());
        }
        opt.step(&mut l2.weight, grads.wrt(&l2.weight).unwrap());
        if let Some(ref mut bias) = l2.bias {
            opt.step(bias, grads.wrt(bias).unwrap());
        }

        if epoch % 100 == 0 {
            println!("Epoch {}: loss = {:.4}", epoch, loss.item());
        }
    }
}
```

The pattern is: forward → loss → backward → step. Each `opt.step()` call mutates the parameter in-place and updates the optimizer's internal state.

## Optimizer Comparison

| Optimizer | State per param | Compute | When to use |
|-----------|----------------|---------|-------------|
| SGD | None | 1 mul, 1 sub | Simple baseline |
| SGD+Momentum | 1 Vec (v) | 3 muls, 2 adds | When SGD oscillates |
| Adam | 2 Vecs (m, v) | 10+ ops | Default choice |
| AdamW | 2 Vecs | 11+ ops | When using weight decay |

Adam uses 3x the memory of SGD but typically converges faster and is less sensitive to learning rate.

## What's Next

We have gradients, we have optimizers. The remaining question: where does the computation happen?

Our `B::mul`, `B::add` operations are abstract. In [Part 4]({% post_url 2026-01-15-deep-learning-scratch-part4-backends %}), we'll implement concrete backends — CPU with SIMD, and Metal for GPU acceleration.

---

*Part 3 of the "Deep Learning from Scratch in Rust" series. [Part 2]({% post_url 2025-11-15-deep-learning-scratch-part2-models %}) covers models and loss, [Part 4]({% post_url 2026-01-15-deep-learning-scratch-part4-backends %}) covers backends.*
