---
layout: post
title:  "Deep Learning in Rust: From Scratch, Part 6 — Pluggable Backends"
date:   2026-02-08 10:00:00 -0700
tags: rust machine-learning programming
author: bolu-atx
categories: programming
---

Throughout this series, we've been writing `B::add`, `B::matmul`, `B::exp` without explaining what `B` actually is. Time to pay that debt.

`B` is a *backend* — an implementation of tensor operations. Different backends can target different hardware:
- CPU with SIMD intrinsics
- Metal shaders for macOS GPUs
- CUDA kernels for NVIDIA GPUs

Today we'll see how Rust's type system lets us write autodiff code once and run it anywhere — with the backend choice resolved entirely at compile time.

<!--more-->

## The Backend Trait

Here's the core abstraction:

```rust
pub trait Backend: Clone + Send + Sync + 'static {
    type Tensor: TensorData;

    // Creation
    fn zeros(shape: &Shape) -> Self::Tensor;
    fn ones(shape: &Shape) -> Self::Tensor;
    fn from_vec(data: Vec<f32>, shape: Shape) -> Self::Tensor;

    // Element-wise ops
    fn add(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn mul(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn exp(x: &Self::Tensor) -> Self::Tensor;
    // ... more ops

    // Linear algebra
    fn matmul(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    // Reductions
    fn sum(x: &Self::Tensor, axes: Option<&[usize]>, keepdims: bool) -> Self::Tensor;
}
```

A backend provides two things:
1. An associated type `Tensor` for storage
2. Static methods implementing every operation

Notice: these are *static* methods, not instance methods. The backend is a type, not a value. This is crucial.

## Type-Level Backend Selection

Our tensor type is parameterized by backend:

```rust
pub struct Tensor<B: Backend>(Arc<TensorNode<B>>);

struct TensorNode<B: Backend> {
    id: NodeId,
    op: TensorOp,
    data: B::Tensor,  // Backend-specific storage!
    children: Vec<Tensor<B>>,
}
```

When you write:

```rust
type T = Tensor<CpuBackend>;

let x = T::var("x", CpuBackend::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3])));
let y = x.exp();
```

The compiler:
1. Sees `Tensor<CpuBackend>`
2. Monomorphizes all methods with `B = CpuBackend`
3. Inlines the backend calls directly

**There's no runtime dispatch.** The backend is "erased" into concrete machine code. This is zero-cost abstraction.

## The CPU Backend

Let's implement a simple CPU backend:

```rust
#[derive(Clone)]
pub struct CpuBackend;

pub struct CpuTensor {
    data: Vec<f32>,
    shape: Shape,
    strides: Strides,
}

impl TensorData for CpuTensor {
    fn shape(&self) -> &Shape { &self.shape }
    fn strides(&self) -> &Strides { &self.strides }
    fn as_slice(&self) -> &[f32] { &self.data }
    fn as_slice_mut(&mut self) -> &mut [f32] { &mut self.data }
}

impl Backend for CpuBackend {
    type Tensor = CpuTensor;

    fn zeros(shape: &Shape) -> CpuTensor {
        CpuTensor {
            data: vec![0.0; shape.numel()],
            shape: shape.clone(),
            strides: shape.contiguous_strides(),
        }
    }

    fn add(a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        // Handle broadcasting...
        let out_shape = broadcast_shape(a.shape(), b.shape()).unwrap();
        let mut out = vec![0.0; out_shape.numel()];

        for i in 0..out.len() {
            let a_idx = broadcast_index(i, &out_shape, a.shape());
            let b_idx = broadcast_index(i, &out_shape, b.shape());
            out[i] = a.data[a_idx] + b.data[b_idx];
        }

        CpuTensor {
            data: out,
            shape: out_shape.clone(),
            strides: out_shape.contiguous_strides(),
        }
    }

    fn exp(x: &CpuTensor) -> CpuTensor {
        let data: Vec<f32> = x.data.iter().map(|&v| v.exp()).collect();
        CpuTensor {
            data,
            shape: x.shape.clone(),
            strides: x.strides.clone(),
        }
    }

    // ... implement all other ops
}
```

This works, but it's slow. Let's add SIMD.

## SIMD Acceleration

Modern CPUs have vector instructions that process 4-8 floats simultaneously:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn add_simd(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let simd_width = 8;  // AVX processes 8 floats
    let simd_end = n - (n % simd_width);

    // SIMD loop
    unsafe {
        for i in (0..simd_end).step_by(simd_width) {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vout = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), vout);
        }
    }

    // Scalar tail
    for i in simd_end..n {
        out[i] = a[i] + b[i];
    }
}
```

For transcendental functions (exp, log, sin), we can use approximations:

```rust
// Fast exp approximation using polynomial (Padé or Taylor)
fn exp_simd(x: &[f32], out: &mut [f32]) {
    unsafe {
        for i in (0..x.len()).step_by(8) {
            let v = _mm256_loadu_ps(x.as_ptr().add(i));

            // Clamp to avoid overflow
            let v = _mm256_min_ps(v, _mm256_set1_ps(88.0));
            let v = _mm256_max_ps(v, _mm256_set1_ps(-88.0));

            // exp(x) ≈ (1 + x/n)^n, using range reduction and polynomial
            // ... polynomial approximation
            let result = fast_exp_avx(v);

            _mm256_storeu_ps(out.as_mut_ptr().add(i), result);
        }
    }
}
```

Speedup: 4-8x for element-wise ops compared to scalar loops.

## Matrix Multiplication: The Critical Op

MatMul dominates neural network compute. A naive implementation:

```rust
fn matmul_naive(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}
```

This is cache-hostile. Better: tile the matrices to fit in L1/L2 cache:

```rust
fn matmul_tiled(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    const TILE: usize = 64;  // Fits in L1 cache

    for i0 in (0..m).step_by(TILE) {
        for j0 in (0..n).step_by(TILE) {
            for l0 in (0..k).step_by(TILE) {
                // Process TILE×TILE block
                for i in i0..min(i0 + TILE, m) {
                    for l in l0..min(l0 + TILE, k) {
                        let a_val = a[i * k + l];
                        for j in j0..min(j0 + TILE, n) {
                            c[i * n + j] += a_val * b[l * n + j];
                        }
                    }
                }
            }
        }
    }
}
```

For production, call BLAS (OpenBLAS, Intel MKL, Accelerate):

```rust
fn matmul_blas(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    unsafe {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m as i32, n as i32, k as i32,
            1.0,                    // alpha
            a.as_ptr(), k as i32,   // A
            b.as_ptr(), n as i32,   // B
            0.0,                    // beta
            c.as_mut_ptr(), n as i32, // C
        );
    }
}
```

BLAS implementations are heavily optimized — hand-tuned assembly, multi-threading, architecture-specific kernels.

## The Metal Backend

On macOS, we can use Metal for GPU acceleration:

```rust
#[derive(Clone)]
pub struct MetalBackend {
    device: metal::Device,
    command_queue: metal::CommandQueue,
}

pub struct MetalTensor {
    buffer: metal::Buffer,
    shape: Shape,
    strides: Strides,
}
```

Metal operations work differently:
1. Data lives in GPU memory (`MTLBuffer`)
2. Operations are encoded into command buffers
3. Work is submitted asynchronously

```rust
impl Backend for MetalBackend {
    type Tensor = MetalTensor;

    fn add(a: &MetalTensor, b: &MetalTensor) -> MetalTensor {
        let device = metal::Device::system_default().unwrap();
        let out_shape = broadcast_shape(a.shape(), b.shape()).unwrap();
        let out_buffer = device.new_buffer(
            (out_shape.numel() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Load compute shader
        let library = device.new_library_with_source(ADD_SHADER, &Default::default()).unwrap();
        let function = library.get_function("add_tensors", None).unwrap();
        let pipeline = device.new_compute_pipeline_state_with_function(&function).unwrap();

        // Encode and dispatch
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&a.buffer), 0);
        encoder.set_buffer(1, Some(&b.buffer), 0);
        encoder.set_buffer(2, Some(&out_buffer), 0);
        // ... set shape uniforms

        let threads_per_group = MTLSize::new(256, 1, 1);
        let num_groups = MTLSize::new(
            (out_shape.numel() as u64 + 255) / 256, 1, 1
        );
        encoder.dispatch_thread_groups(num_groups, threads_per_group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        MetalTensor {
            buffer: out_buffer,
            shape: out_shape,
            strides: out_shape.contiguous_strides(),
        }
    }
}
```

The Metal shader (in MSL):

```cpp
#include <metal_stdlib>
using namespace metal;

kernel void add_tensors(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    // Handle broadcasting in shader...
    out[idx] = a[idx] + b[idx];
}
```

## Compile-Time Backend Selection

The magic of Rust's type system: backend choice happens at compile time.

```rust
// Choose backend once at the top
#[cfg(feature = "metal")]
type B = MetalBackend;

#[cfg(not(feature = "metal"))]
type B = CpuBackend;

type T = Tensor<B>;

// All downstream code is generic
fn train_model(inputs: &T, targets: &T) -> T {
    // This code works for ANY backend
    let hidden = inputs.matmul(&weights) + &biases;
    let output = hidden.relu();
    let loss = mse_loss(&output, targets);
    loss.backward()
    // ...
}
```

You can also make it runtime-selectable, but you pay the cost of dynamic dispatch:

```rust
enum AnyBackend {
    Cpu(CpuBackend),
    Metal(MetalBackend),
}

// Now operations require match statements or trait objects
// This adds overhead, but sometimes worth it for flexibility
```

## Backend Interop

Sometimes you want to move data between backends:

```rust
trait IntoBackend<Target: Backend>: Backend {
    fn convert(tensor: &Self::Tensor) -> Target::Tensor;
}

impl IntoBackend<MetalBackend> for CpuBackend {
    fn convert(tensor: &CpuTensor) -> MetalTensor {
        // Copy CPU data to GPU buffer
        let device = metal::Device::system_default().unwrap();
        let buffer = device.new_buffer_with_data(
            tensor.data.as_ptr() as *const _,
            (tensor.data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        MetalTensor {
            buffer,
            shape: tensor.shape.clone(),
            strides: tensor.strides.clone(),
        }
    }
}
```

## When to Use Which Backend

| Backend | Best for | Latency | Throughput |
|---------|----------|---------|------------|
| CPU (scalar) | Debugging, small tensors | Low | Low |
| CPU (SIMD) | Medium tensors, dev machines | Low | Medium |
| Metal | macOS, large batches | Medium | High |
| CUDA | NVIDIA GPUs, production | Medium | Highest |

Rules of thumb:
- Tensors < 1000 elements: CPU is fine
- Tensors < 100K elements: SIMD CPU competitive with GPU
- Tensors > 100K elements: GPU wins for throughput
- Latency-sensitive: CPU avoids kernel launch overhead

## The Power of Zero-Cost Abstraction

Here's what's remarkable: our autodiff code doesn't know or care about backends.

```rust
// This is backend-agnostic
pub fn backward<B: Backend>(output: &Tensor<B>) -> Gradients<B> {
    // Same algorithm, regardless of B
}

// So is the optimizer
impl<B: Backend> Optimizer<B> for Adam<B> {
    fn step(&mut self, ...) -> B::Tensor {
        // Uses B::mul, B::add, etc.
    }
}
```

When you compile with `--features metal`, the entire autodiff engine becomes Metal-accelerated. When you compile for CPU, it uses SIMD. Same source code, optimal machine code for each target.

This is Rust's promise: write generic, get specialized.

## Summary

We've covered:
1. **Backend trait**: Abstract interface for tensor operations
2. **Type-level selection**: Backend resolved at compile time, zero runtime cost
3. **CPU backend**: Scalar loops → SIMD → BLAS
4. **Metal backend**: GPU shaders for parallel compute
5. **Interop**: Moving data between backends

The autodiff engine from Parts 1-5 runs on any backend that implements the trait. Write once, run anywhere — at native speed.

---

*This concludes the "Deep Learning in Rust: From Scratch" series. From scalar intuition to tensor gradients to models to optimizers to hardware backends — we've built a working ML framework.*

*Full source code: [github.com/bolu-atx/autodiff-rs](https://github.com/bolu-atx/autodiff-rs)*
