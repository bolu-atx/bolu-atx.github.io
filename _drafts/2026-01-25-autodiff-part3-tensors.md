---
layout: post
title:  "Understanding Autodiff, Part 3: The Tensor Generalization"
date:   2026-01-25 10:00:00 -0700
tags: rust machine-learning programming
author: bolu-atx
categories: programming
---

In [Part 2]({% post_url 2026-01-17-autodiff-part2-implementation %}), we built a working autodiff engine for scalar functions. Clean, elegant, and... completely impractical.

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

Not quite. Three new problems emerge.

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

We have tensors with gradients. To actually train a model, we need:
- **Loss functions**: How wrong is our prediction?
- **Optimizers**: How do we update parameters using gradients?

[Part 4]({% post_url 2026-02-01-autodiff-part4-optimizers %}) implements SGD and Adam.

---

*Part 3 of a series on building autodiff from scratch. [Part 1]({% post_url 2026-01-03-autodiff-part1-intuition %}) covers intuition, [Part 2]({% post_url 2026-01-17-autodiff-part2-implementation %}) builds a scalar engine.*
