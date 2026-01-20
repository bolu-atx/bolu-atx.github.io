---
layout: post
title:  "Deep Learning in Rust: From Scratch, Part 5 — Optimizers"
date:   2026-02-01 10:00:00 -0700
tags: rust machine-learning programming
author: bolu-atx
categories: programming
---

We have gradients. Now what?

In [Part 4]({% post_url 2026-01-29-deep-learning-rust-part4-models %}), we built layers, models, and loss functions. Given a model and a loss, autodiff computes ∂loss/∂θ for every parameter θ. But gradients alone don't train a model. We need an *optimizer* to turn gradients into parameter updates.

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

Watch SGD struggle on a narrow valley — the classic pathological case:

<div id="optimizer-2d-viz" style="width: 100%; max-width: 700px; margin: 2rem auto;"></div>
<div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem;">
  <button id="reset-optimizers" style="padding: 0.5rem 1rem; cursor: pointer;">Reset</button>
  <label style="display: flex; align-items: center; gap: 0.5rem;">
    Learning Rate: <input type="range" id="lr-slider" min="0.001" max="0.1" step="0.001" value="0.02">
    <span id="lr-value">0.020</span>
  </label>
</div>
<div style="display: flex; justify-content: center; gap: 2rem; font-size: 0.9rem; margin-bottom: 2rem;">
  <span><span style="display: inline-block; width: 12px; height: 12px; background: #e63946; border-radius: 50%;"></span> SGD</span>
  <span><span style="display: inline-block; width: 12px; height: 12px; background: #2a9d8f; border-radius: 50%;"></span> Momentum</span>
  <span><span style="display: inline-block; width: 12px; height: 12px; background: #e9c46a; border-radius: 50%;"></span> Adam</span>
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
(function() {
  const width = 700, height = 500;
  const margin = { top: 20, right: 20, bottom: 40, left: 50 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  // Beale function - classic narrow valley optimization test
  function beale(x, y) {
    const a = 1.5 - x + x * y;
    const b = 2.25 - x + x * y * y;
    const c = 2.625 - x + x * y * y * y;
    return a * a + b * b + c * c;
  }

  // Gradient of Beale function
  function bealeGrad(x, y) {
    const a = 1.5 - x + x * y;
    const b = 2.25 - x + x * y * y;
    const c = 2.625 - x + x * y * y * y;
    const dx = 2 * a * (-1 + y) + 2 * b * (-1 + y * y) + 2 * c * (-1 + y * y * y);
    const dy = 2 * a * x + 2 * b * 2 * x * y + 2 * c * 3 * x * y * y;
    return [dx, dy];
  }

  const xDomain = [-1, 4.5];
  const yDomain = [-1, 2];
  const xScale = d3.scaleLinear().domain(xDomain).range([0, plotWidth]);
  const yScale = d3.scaleLinear().domain(yDomain).range([plotHeight, 0]);

  // Generate contour data
  const n = 200;
  const values = new Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const x = xDomain[0] + (xDomain[1] - xDomain[0]) * i / (n - 1);
      const y = yDomain[0] + (yDomain[1] - yDomain[0]) * j / (n - 1);
      values[j * n + i] = Math.log(1 + beale(x, y));
    }
  }

  const contours = d3.contours()
    .size([n, n])
    .thresholds(d3.range(0, 8, 0.3))
    (values);

  const container = d3.select("#optimizer-2d-viz");
  const svg = container.append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("preserveAspectRatio", "xMidYMid meet")
    .style("width", "100%")
    .style("height", "auto")
    .style("background", "var(--bg-primary, #1a1a2e)");

  const g = svg.append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // Color scale for contours
  const color = d3.scaleSequential(d3.interpolateViridis).domain([0, 8]);

  // Transform contours to plot coordinates
  const transform = d3.geoTransform({
    point: function(x, y) {
      this.stream.point(
        xScale(xDomain[0] + (xDomain[1] - xDomain[0]) * x / (n - 1)),
        yScale(yDomain[0] + (yDomain[1] - yDomain[0]) * y / (n - 1))
      );
    }
  });
  const path = d3.geoPath(transform);

  g.selectAll("path.contour")
    .data(contours)
    .enter().append("path")
    .attr("class", "contour")
    .attr("d", path)
    .attr("fill", d => color(d.value))
    .attr("stroke", "rgba(255,255,255,0.1)")
    .attr("stroke-width", 0.5);

  // Mark the minimum
  g.append("circle")
    .attr("cx", xScale(3))
    .attr("cy", yScale(0.5))
    .attr("r", 6)
    .attr("fill", "none")
    .attr("stroke", "#fff")
    .attr("stroke-width", 2);

  // Axes
  g.append("g")
    .attr("transform", `translate(0,${plotHeight})`)
    .call(d3.axisBottom(xScale).ticks(6))
    .attr("color", "var(--text-secondary, #888)");

  g.append("g")
    .call(d3.axisLeft(yScale).ticks(4))
    .attr("color", "var(--text-secondary, #888)");

  g.append("text")
    .attr("x", plotWidth / 2)
    .attr("y", plotHeight + 35)
    .attr("fill", "var(--text-primary, #fff)")
    .attr("text-anchor", "middle")
    .text("θ₁");

  g.append("text")
    .attr("transform", "rotate(-90)")
    .attr("x", -plotHeight / 2)
    .attr("y", -35)
    .attr("fill", "var(--text-primary, #fff)")
    .attr("text-anchor", "middle")
    .text("θ₂");

  // Optimizer state
  let lr = 0.02;
  const startPos = [0, 1.5];

  class SGDOptimizer {
    constructor() { this.reset(); }
    reset() { this.pos = [...startPos]; this.path = [[...this.pos]]; }
    step(lr) {
      const [gx, gy] = bealeGrad(this.pos[0], this.pos[1]);
      const gradNorm = Math.sqrt(gx*gx + gy*gy);
      const clippedNorm = Math.min(gradNorm, 50);
      const scale = gradNorm > 0 ? clippedNorm / gradNorm : 0;
      this.pos[0] -= lr * gx * scale;
      this.pos[1] -= lr * gy * scale;
      this.path.push([...this.pos]);
    }
  }

  class MomentumOptimizer {
    constructor() { this.reset(); }
    reset() { this.pos = [...startPos]; this.v = [0, 0]; this.path = [[...this.pos]]; }
    step(lr) {
      const [gx, gy] = bealeGrad(this.pos[0], this.pos[1]);
      const gradNorm = Math.sqrt(gx*gx + gy*gy);
      const clippedNorm = Math.min(gradNorm, 50);
      const scale = gradNorm > 0 ? clippedNorm / gradNorm : 0;
      this.v[0] = 0.9 * this.v[0] + gx * scale;
      this.v[1] = 0.9 * this.v[1] + gy * scale;
      this.pos[0] -= lr * this.v[0];
      this.pos[1] -= lr * this.v[1];
      this.path.push([...this.pos]);
    }
  }

  class AdamOptimizer {
    constructor() { this.reset(); }
    reset() {
      this.pos = [...startPos];
      this.m = [0, 0];
      this.v = [0, 0];
      this.t = 0;
      this.path = [[...this.pos]];
    }
    step(lr) {
      this.t++;
      const [gx, gy] = bealeGrad(this.pos[0], this.pos[1]);
      const gradNorm = Math.sqrt(gx*gx + gy*gy);
      const clippedNorm = Math.min(gradNorm, 50);
      const scale = gradNorm > 0 ? clippedNorm / gradNorm : 0;
      const cgx = gx * scale, cgy = gy * scale;

      this.m[0] = 0.9 * this.m[0] + 0.1 * cgx;
      this.m[1] = 0.9 * this.m[1] + 0.1 * cgy;
      this.v[0] = 0.999 * this.v[0] + 0.001 * cgx * cgx;
      this.v[1] = 0.999 * this.v[1] + 0.001 * cgy * cgy;

      const bc1 = 1 - Math.pow(0.9, this.t);
      const bc2 = 1 - Math.pow(0.999, this.t);
      const mHat = [this.m[0] / bc1, this.m[1] / bc1];
      const vHat = [this.v[0] / bc2, this.v[1] / bc2];

      this.pos[0] -= lr * mHat[0] / (Math.sqrt(vHat[0]) + 1e-8);
      this.pos[1] -= lr * mHat[1] / (Math.sqrt(vHat[1]) + 1e-8);
      this.path.push([...this.pos]);
    }
  }

  const sgd = new SGDOptimizer();
  const momentum = new MomentumOptimizer();
  const adam = new AdamOptimizer();

  const line = d3.line()
    .x(d => xScale(d[0]))
    .y(d => yScale(d[1]));

  const sgdPath = g.append("path").attr("fill", "none").attr("stroke", "#e63946").attr("stroke-width", 2);
  const momPath = g.append("path").attr("fill", "none").attr("stroke", "#2a9d8f").attr("stroke-width", 2);
  const adamPath = g.append("path").attr("fill", "none").attr("stroke", "#e9c46a").attr("stroke-width", 2);

  const sgdDot = g.append("circle").attr("r", 5).attr("fill", "#e63946");
  const momDot = g.append("circle").attr("r", 5).attr("fill", "#2a9d8f");
  const adamDot = g.append("circle").attr("r", 5).attr("fill", "#e9c46a");

  function updatePaths() {
    sgdPath.attr("d", line(sgd.path));
    momPath.attr("d", line(momentum.path));
    adamPath.attr("d", line(adam.path));
    sgdDot.attr("cx", xScale(sgd.pos[0])).attr("cy", yScale(sgd.pos[1]));
    momDot.attr("cx", xScale(momentum.pos[0])).attr("cy", yScale(momentum.pos[1]));
    adamDot.attr("cx", xScale(adam.pos[0])).attr("cy", yScale(adam.pos[1]));
  }

  updatePaths();

  let animating = true;
  let stepCount = 0;
  const maxSteps = 300;

  function animate() {
    if (animating && stepCount < maxSteps) {
      sgd.step(lr);
      momentum.step(lr);
      adam.step(lr);
      stepCount++;
      updatePaths();
    }
    requestAnimationFrame(animate);
  }
  animate();

  document.getElementById("reset-optimizers").addEventListener("click", () => {
    sgd.reset();
    momentum.reset();
    adam.reset();
    stepCount = 0;
    animating = true;
    updatePaths();
  });

  const lrSlider = document.getElementById("lr-slider");
  const lrValue = document.getElementById("lr-value");
  lrSlider.addEventListener("input", () => {
    lr = parseFloat(lrSlider.value);
    lrValue.textContent = lr.toFixed(3);
  });
})();
</script>

Notice how SGD oscillates back and forth across the narrow valley, making slow progress. Momentum builds up speed in the consistent direction while dampening oscillations. Adam adapts — taking larger steps where the gradient is stable.

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

### The Race in 3D

Here's the same optimization viewed as balls rolling down a loss surface. Drag to rotate, scroll to zoom:

<div id="optimizer-3d-viz" style="width: 100%; max-width: 800px; height: 500px; margin: 2rem auto; border-radius: 8px; overflow: hidden;"></div>
<div style="display: flex; justify-content: center; gap: 1rem; margin-bottom: 2rem;">
  <button id="reset-3d" style="padding: 0.5rem 1rem; cursor: pointer;">Reset Race</button>
  <button id="toggle-trails-3d" style="padding: 0.5rem 1rem; cursor: pointer;">Toggle Trails</button>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function() {
  const container = document.getElementById('optimizer-3d-viz');
  if (!container) return;

  const width = container.clientWidth || 800;
  const height = 500;

  // Scene setup
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
  camera.position.set(8, 6, 8);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(ambientLight);
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(5, 10, 5);
  scene.add(directionalLight);

  // Loss function (simplified Beale-like with visible minimum)
  function loss(x, z) {
    const a = 1.5 - x + x * z;
    const b = 2.25 - x + x * z * z;
    return 0.15 * (a * a + b * b);
  }

  function lossGrad(x, z) {
    const a = 1.5 - x + x * z;
    const b = 2.25 - x + x * z * z;
    const dx = 0.15 * 2 * (a * (-1 + z) + b * (-1 + z * z));
    const dz = 0.15 * 2 * (a * x + b * 2 * x * z);
    return [dx, dz];
  }

  // Create surface geometry
  const surfaceSize = 8;
  const surfaceRes = 80;
  const geometry = new THREE.PlaneGeometry(surfaceSize, surfaceSize, surfaceRes, surfaceRes);
  geometry.rotateX(-Math.PI / 2);

  const positions = geometry.attributes.position;
  const colors = [];
  const colorScale = new THREE.Color();

  for (let i = 0; i < positions.count; i++) {
    const x = positions.getX(i);
    const z = positions.getZ(i);
    const y = loss(x, z);
    positions.setY(i, y);

    // Color by height
    const t = Math.min(y / 3, 1);
    colorScale.setHSL(0.7 - t * 0.5, 0.7, 0.4 + t * 0.3);
    colors.push(colorScale.r, colorScale.g, colorScale.b);
  }

  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.computeVertexNormals();

  const surfaceMaterial = new THREE.MeshStandardMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    flatShading: false,
    transparent: true,
    opacity: 0.85,
  });
  const surface = new THREE.Mesh(geometry, surfaceMaterial);
  scene.add(surface);

  // Wireframe overlay
  const wireframe = new THREE.LineSegments(
    new THREE.WireframeGeometry(geometry),
    new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.1, transparent: true })
  );
  scene.add(wireframe);

  // Mark minimum (at approximately x=3, z=0.5)
  const minMarker = new THREE.Mesh(
    new THREE.RingGeometry(0.15, 0.25, 32),
    new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide })
  );
  minMarker.rotation.x = -Math.PI / 2;
  minMarker.position.set(3, loss(3, 0.5) + 0.05, 0.5);
  scene.add(minMarker);

  // Optimizer balls
  const startPos = [-2, 2];
  const ballRadius = 0.12;

  function createBall(color) {
    const ball = new THREE.Mesh(
      new THREE.SphereGeometry(ballRadius, 16, 16),
      new THREE.MeshStandardMaterial({ color, metalness: 0.3, roughness: 0.4 })
    );
    return ball;
  }

  const sgdBall = createBall(0xe63946);
  const momBall = createBall(0x2a9d8f);
  const adamBall = createBall(0xe9c46a);
  scene.add(sgdBall, momBall, adamBall);

  // Trail lines
  let showTrails = true;
  const trailMaterial = {
    sgd: new THREE.LineBasicMaterial({ color: 0xe63946, linewidth: 2 }),
    mom: new THREE.LineBasicMaterial({ color: 0x2a9d8f, linewidth: 2 }),
    adam: new THREE.LineBasicMaterial({ color: 0xe9c46a, linewidth: 2 }),
  };
  let sgdTrail, momTrail, adamTrail;

  // Optimizer states
  class Optimizer3D {
    constructor(type) {
      this.type = type;
      this.reset();
    }
    reset() {
      this.pos = [...startPos];
      this.v = [0, 0];
      this.m = [0, 0];
      this.vAdam = [0, 0];
      this.t = 0;
      this.path = [[...this.pos]];
    }
    step(lr = 0.03) {
      const [gx, gz] = lossGrad(this.pos[0], this.pos[1]);
      const gradNorm = Math.sqrt(gx*gx + gz*gz);
      const clippedNorm = Math.min(gradNorm, 20);
      const scale = gradNorm > 0 ? clippedNorm / gradNorm : 0;
      const cgx = gx * scale, cgz = gz * scale;

      if (this.type === 'sgd') {
        this.pos[0] -= lr * cgx;
        this.pos[1] -= lr * cgz;
      } else if (this.type === 'momentum') {
        this.v[0] = 0.9 * this.v[0] + cgx;
        this.v[1] = 0.9 * this.v[1] + cgz;
        this.pos[0] -= lr * this.v[0];
        this.pos[1] -= lr * this.v[1];
      } else if (this.type === 'adam') {
        this.t++;
        this.m[0] = 0.9 * this.m[0] + 0.1 * cgx;
        this.m[1] = 0.9 * this.m[1] + 0.1 * cgz;
        this.vAdam[0] = 0.999 * this.vAdam[0] + 0.001 * cgx * cgx;
        this.vAdam[1] = 0.999 * this.vAdam[1] + 0.001 * cgz * cgz;
        const bc1 = 1 - Math.pow(0.9, this.t);
        const bc2 = 1 - Math.pow(0.999, this.t);
        const mHat = [this.m[0] / bc1, this.m[1] / bc1];
        const vHat = [this.vAdam[0] / bc2, this.vAdam[1] / bc2];
        this.pos[0] -= lr * mHat[0] / (Math.sqrt(vHat[0]) + 1e-8);
        this.pos[1] -= lr * mHat[1] / (Math.sqrt(vHat[1]) + 1e-8);
      }
      this.path.push([...this.pos]);
    }
  }

  const sgd = new Optimizer3D('sgd');
  const mom = new Optimizer3D('momentum');
  const adam = new Optimizer3D('adam');

  function updateBallPosition(ball, pos) {
    ball.position.set(pos[0], loss(pos[0], pos[1]) + ballRadius, pos[1]);
  }

  function createTrailGeometry(path) {
    const points = path.map(p => new THREE.Vector3(p[0], loss(p[0], p[1]) + 0.02, p[1]));
    return new THREE.BufferGeometry().setFromPoints(points);
  }

  function updateTrails() {
    if (sgdTrail) scene.remove(sgdTrail);
    if (momTrail) scene.remove(momTrail);
    if (adamTrail) scene.remove(adamTrail);

    if (showTrails) {
      sgdTrail = new THREE.Line(createTrailGeometry(sgd.path), trailMaterial.sgd);
      momTrail = new THREE.Line(createTrailGeometry(mom.path), trailMaterial.mom);
      adamTrail = new THREE.Line(createTrailGeometry(adam.path), trailMaterial.adam);
      scene.add(sgdTrail, momTrail, adamTrail);
    }
  }

  function resetAll() {
    sgd.reset();
    mom.reset();
    adam.reset();
    stepCount = 0;
    animating = true;
    updateBallPosition(sgdBall, sgd.pos);
    updateBallPosition(momBall, mom.pos);
    updateBallPosition(adamBall, adam.pos);
    updateTrails();
  }

  updateBallPosition(sgdBall, sgd.pos);
  updateBallPosition(momBall, mom.pos);
  updateBallPosition(adamBall, adam.pos);

  let animating = true;
  let stepCount = 0;
  const maxSteps = 250;

  function animate() {
    requestAnimationFrame(animate);

    if (animating && stepCount < maxSteps) {
      sgd.step();
      mom.step();
      adam.step();
      stepCount++;

      updateBallPosition(sgdBall, sgd.pos);
      updateBallPosition(momBall, mom.pos);
      updateBallPosition(adamBall, adam.pos);
      updateTrails();
    }

    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  document.getElementById('reset-3d').addEventListener('click', resetAll);
  document.getElementById('toggle-trails-3d').addEventListener('click', () => {
    showTrails = !showTrails;
    updateTrails();
  });

  // Handle resize
  window.addEventListener('resize', () => {
    const newWidth = container.clientWidth || 800;
    camera.aspect = newWidth / height;
    camera.updateProjectionMatrix();
    renderer.setSize(newWidth, height);
  });
})();
</script>

The 3D view makes it visceral: SGD (red) zig-zags down the valley walls. Momentum (green) overshoots but self-corrects. Adam (yellow) finds the efficient path.

## What's Next

We have gradients, we have optimizers. The remaining question: where does the computation happen?

Our `B::mul`, `B::add` operations are abstract. In [Part 6]({% post_url 2026-02-08-deep-learning-rust-part6-backends %}), we'll implement concrete backends — CPU with SIMD, and Metal for GPU acceleration.

---

*Part 5 of the "Deep Learning in Rust: From Scratch" series. [Part 4]({% post_url 2026-01-29-deep-learning-rust-part4-models %}) covers models and loss, [Part 6]({% post_url 2026-02-08-deep-learning-rust-part6-backends %}) covers backends.*
