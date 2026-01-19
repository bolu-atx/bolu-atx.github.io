---
layout: post
title:  "Understanding Autodiff, Part 4: SGD, Momentum, and Adam"
date:   2026-02-01 10:00:00 -0700
tags: rust machine-learning programming
author: bolu-atx
categories: programming
---

We have gradients. Now what?

In [Part 3]({% post_url 2026-01-25-autodiff-part3-tensors %}), we built tensor autodiff — given a loss function, we can compute ∂loss/∂θ for every parameter θ. But gradients alone don't train a model. We need an *optimizer* to turn gradients into parameter updates.

Today we'll implement the three most important optimizers: SGD, SGD with Momentum, and Adam. Along the way, we'll see why Adam became the default choice.

<!--more-->

## The Optimization Problem

Training a neural network means minimizing a loss function:

$$\theta^* = \arg\min_\theta \mathcal{L}(\theta)$$

Where θ represents all trainable parameters (weights, biases). The gradient ∇L(θ) points in the direction of steepest *increase*, so we move in the opposite direction:

$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$$

This is gradient descent. The learning rate η controls step size.

## The Optimizer Trait

Let's define what an optimizer does:

```rust
pub trait Optimizer<B: Backend> {
    /// Initialize state for a parameter (called once per parameter).
    fn init_state(&mut self, param_id: NodeId, shape: &Shape);

    /// Compute update and apply to parameter.
    /// Returns the updated parameter data.
    fn step(&mut self, param_id: NodeId, param: &B::Tensor, grad: &B::Tensor) -> B::Tensor;

    /// Zero all accumulated gradients (for next iteration).
    fn zero_grad(&mut self);
}
```

Each optimizer:
1. May maintain per-parameter state (momentum buffers, variance estimates)
2. Takes current parameters and gradients, returns updated parameters
3. Resets between training iterations

## SGD: The Simplest Optimizer

Stochastic Gradient Descent is the baseline:

$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$

Where $g_t = \nabla \mathcal{L}(\theta_t)$ is the gradient.

```rust
pub struct SGD {
    lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        SGD { lr }
    }
}

impl<B: Backend> Optimizer<B> for SGD {
    fn init_state(&mut self, _param_id: NodeId, _shape: &Shape) {
        // SGD has no per-parameter state
    }

    fn step(&mut self, _param_id: NodeId, param: &B::Tensor, grad: &B::Tensor) -> B::Tensor {
        // θ = θ - lr * grad
        let scaled_grad = B::mul(&B::scalar(self.lr), grad);
        B::sub(param, &scaled_grad)
    }

    fn zero_grad(&mut self) {}
}
```

SGD is simple but has problems:
- Gets stuck in flat regions (gradient ≈ 0)
- Oscillates in narrow valleys
- Learning rate is sensitive — too high diverges, too low is slow

## SGD with Momentum

Momentum adds "velocity" to the parameter updates:

$$v_{t+1} = \mu \cdot v_t + g_t$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_{t+1}$$

The velocity accumulates gradients over time. If gradients consistently point the same direction, velocity builds up. If they oscillate, they cancel out.

```rust
pub struct SGDMomentum<B: Backend> {
    lr: f32,
    momentum: f32,
    velocities: HashMap<NodeId, B::Tensor>,
}

impl<B: Backend> SGDMomentum<B> {
    pub fn new(lr: f32, momentum: f32) -> Self {
        SGDMomentum {
            lr,
            momentum,
            velocities: HashMap::new(),
        }
    }
}

impl<B: Backend> Optimizer<B> for SGDMomentum<B> {
    fn init_state(&mut self, param_id: NodeId, shape: &Shape) {
        // Initialize velocity to zeros
        self.velocities.insert(param_id, B::zeros(shape));
    }

    fn step(&mut self, param_id: NodeId, param: &B::Tensor, grad: &B::Tensor) -> B::Tensor {
        let v = self.velocities.get_mut(&param_id).unwrap();

        // v = momentum * v + grad
        let momentum_v = B::mul(&B::scalar(self.momentum), v);
        *v = B::add(&momentum_v, grad);

        // θ = θ - lr * v
        let scaled_v = B::mul(&B::scalar(self.lr), v);
        B::sub(param, &scaled_v)
    }

    fn zero_grad(&mut self) {}
}
```

Typical momentum value: 0.9 (90% of previous velocity retained).

### Nesterov Momentum (NAG)

A small tweak that often works better — compute gradient at the "lookahead" position:

$$v_{t+1} = \mu \cdot v_t + \nabla \mathcal{L}(\theta_t - \mu \cdot v_t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_{t+1}$$

```rust
fn step(&mut self, param_id: NodeId, param: &B::Tensor, grad: &B::Tensor) -> B::Tensor {
    let v = self.velocities.get_mut(&param_id).unwrap();

    // v_new = momentum * v + grad
    let v_new = B::add(&B::mul(&B::scalar(self.momentum), v), grad);

    // θ = θ - lr * (momentum * v_new + grad)
    // This is the "Nesterov trick" applied after the fact
    let nesterov_grad = B::add(&B::mul(&B::scalar(self.momentum), &v_new), grad);
    let update = B::mul(&B::scalar(self.lr), &nesterov_grad);

    *v = v_new;
    B::sub(param, &update)
}
```

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
pub struct Adam<B: Backend> {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,  // Timestep for bias correction
    m: HashMap<NodeId, B::Tensor>,  // First moment
    v: HashMap<NodeId, B::Tensor>,  // Second moment
}

impl<B: Backend> Adam<B> {
    pub fn new(lr: f32) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
}

impl<B: Backend> Optimizer<B> for Adam<B> {
    fn init_state(&mut self, param_id: NodeId, shape: &Shape) {
        self.m.insert(param_id, B::zeros(shape));
        self.v.insert(param_id, B::zeros(shape));
    }

    fn step(&mut self, param_id: NodeId, param: &B::Tensor, grad: &B::Tensor) -> B::Tensor {
        self.t += 1;  // Increment timestep

        let m = self.m.get_mut(&param_id).unwrap();
        let v = self.v.get_mut(&param_id).unwrap();

        // m = β₁ * m + (1 - β₁) * g
        let beta1_m = B::mul(&B::scalar(self.beta1), m);
        let one_minus_beta1_g = B::mul(&B::scalar(1.0 - self.beta1), grad);
        *m = B::add(&beta1_m, &one_minus_beta1_g);

        // v = β₂ * v + (1 - β₂) * g²
        let beta2_v = B::mul(&B::scalar(self.beta2), v);
        let g_squared = B::mul(grad, grad);
        let one_minus_beta2_g2 = B::mul(&B::scalar(1.0 - self.beta2), &g_squared);
        *v = B::add(&beta2_v, &one_minus_beta2_g2);

        // Bias correction
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        let m_hat = B::div(m, &B::scalar(bias_correction1));
        let v_hat = B::div(v, &B::scalar(bias_correction2));

        // θ = θ - lr * m_hat / (sqrt(v_hat) + ε)
        let sqrt_v_hat = B::sqrt(&v_hat);
        let denom = B::add(&sqrt_v_hat, &B::scalar(self.eps));
        let update = B::div(&m_hat, &denom);
        let scaled_update = B::mul(&B::scalar(self.lr), &update);

        B::sub(param, &scaled_update)
    }

    fn zero_grad(&mut self) {
        // Note: Adam doesn't zero m and v between iterations
        // They accumulate across the entire training run
    }
}
```

### Why Adam Works

1. **Momentum from m**: Smooths out gradient noise
2. **Adaptivity from v**: Learns different rates per parameter
3. **Bias correction**: Prevents early updates from being too small

Default hyperparameters (β₁=0.9, β₂=0.999, ε=1e-8) work well for most problems.

## AdamW: Weight Decay Done Right

Original Adam has a subtle bug with L2 regularization. Weight decay should shrink parameters toward zero:

$$\theta_{t+1} = (1 - \lambda) \cdot \theta_t - \eta \cdot \text{update}$$

But if you add L2 to the loss, Adam's adaptive scaling weakens the regularization effect. AdamW fixes this by decoupling weight decay:

```rust
pub struct AdamW<B: Backend> {
    // ... same as Adam
    weight_decay: f32,
}

impl<B: Backend> Optimizer<B> for AdamW<B> {
    fn step(&mut self, param_id: NodeId, param: &B::Tensor, grad: &B::Tensor) -> B::Tensor {
        // ... compute m_hat, v_hat same as Adam

        // Apply weight decay BEFORE the Adam update
        let decayed_param = B::mul(&B::scalar(1.0 - self.lr * self.weight_decay), param);

        // Then apply Adam update
        let update = B::div(&m_hat, &B::add(&B::sqrt(&v_hat), &B::scalar(self.eps)));
        let scaled_update = B::mul(&B::scalar(self.lr), &update);

        B::sub(&decayed_param, &scaled_update)
    }
}
```

## The Training Loop

Putting it all together:

```rust
fn train_step<B: Backend>(
    model: &Model<B>,
    optimizer: &mut impl Optimizer<B>,
    inputs: &Tensor<B>,
    targets: &Tensor<B>,
) -> f32 {
    // 1. Forward pass
    let predictions = model.forward(inputs);

    // 2. Compute loss
    let loss = mse_loss(&predictions, targets);

    // 3. Backward pass
    let grads = loss.backward();

    // 4. Update parameters
    for (param_id, param) in model.parameters() {
        if let Some(grad) = grads.wrt(param) {
            let new_value = optimizer.step(param_id, param.data(), grad);
            param.set_data(new_value);
        }
    }

    loss.item()
}

// Training loop
for epoch in 0..num_epochs {
    for (inputs, targets) in data_loader {
        let loss = train_step(&model, &mut optimizer, &inputs, &targets);
        println!("Loss: {:.4}", loss);
    }
}
```

## Loss Functions

Common losses and their gradients:

### Mean Squared Error

$$\mathcal{L} = \frac{1}{n} \sum_i (y_i - \hat{y}_i)^2$$

```rust
fn mse_loss<B: Backend>(pred: &Tensor<B>, target: &Tensor<B>) -> Tensor<B> {
    let diff = pred - target;
    (&diff * &diff).mean(None, false)
}
```

### Cross-Entropy (for classification)

$$\mathcal{L} = -\frac{1}{n} \sum_i \sum_c y_{ic} \log(\hat{y}_{ic})$$

```rust
fn cross_entropy_loss<B: Backend>(logits: &Tensor<B>, targets: &Tensor<B>) -> Tensor<B> {
    // Softmax for numerical stability
    let max_logits = logits.max(Some(&[-1]), true);
    let shifted = logits - &max_logits;
    let exp_logits = shifted.exp();
    let sum_exp = exp_logits.sum(Some(&[-1]), true);
    let log_probs = &shifted - &sum_exp.log();

    // Negative log likelihood
    let nll = -(targets * &log_probs);
    nll.sum(Some(&[-1]), false).mean(None, false)
}
```

## Learning Rate Schedules

Fixed learning rates rarely work well. Common schedules:

```rust
pub trait LRScheduler {
    fn get_lr(&self, step: usize) -> f32;
}

// Constant
pub struct ConstantLR(f32);

// Step decay: lr = lr₀ * γ^(step / step_size)
pub struct StepLR {
    initial_lr: f32,
    step_size: usize,
    gamma: f32,
}

// Cosine annealing: smooth decay to 0
pub struct CosineAnnealingLR {
    initial_lr: f32,
    total_steps: usize,
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self, step: usize) -> f32 {
        let progress = step as f32 / self.total_steps as f32;
        self.initial_lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}
```

## Optimizer Comparison

| Optimizer | State per param | Compute | When to use |
|-----------|----------------|---------|-------------|
| SGD | None | 1 mul, 1 sub | Simple baseline |
| SGD+Momentum | 1 tensor (v) | 3 muls, 2 adds | When SGD oscillates |
| Adam | 2 tensors (m, v) | 10+ ops | Default choice |
| AdamW | 2 tensors | 11+ ops | When using weight decay |

Adam uses 3x the memory of SGD but typically converges faster and is less sensitive to learning rate.

## What's Next

We have gradients, we have optimizers. The remaining question: where does the computation happen?

Our `B::mul`, `B::add` operations are abstract. In [Part 5]({% post_url 2026-02-08-autodiff-part5-backends %}), we'll implement concrete backends — CPU with SIMD, and Metal for GPU acceleration.

---

*Part 4 of a series on building autodiff from scratch. [Part 3]({% post_url 2026-01-25-autodiff-part3-tensors %}) covers tensor generalization.*
