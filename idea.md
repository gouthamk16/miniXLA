# MiniXLA

> A lightweight ML compiler and runtime for exploring graph compilation, optimization, and PTX code generation on NVIDIA GPUs.

---

## Vision

MiniXLA explores the core ideas behind systems like XLA, TensorRT, TVM, and custom AI infrastructure stacks. The objective is not to compete with PyTorch or JAX, but to understand how high-level tensor operations are transformed into optimized GPU execution through graph compilation, optimization, and PTX code generation.

---

## Goals

1. Build a computational graph representation for tensor operations.
2. Implement graph optimization passes.
3. Generate PTX code directly from optimized graphs.
4. Execute generated kernels on NVIDIA GPUs.
5. Develop a foundation for future work in distributed AI systems.

---

## High-Level Architecture

```
User Operations
       │
       ▼
Computational Graph
       │
       ▼
Graph Optimizer
       │
       ▼
PTX Code Generator
       │
       ▼
GPU Runtime
       │
       ▼
Execution Results
```

---

## Phases

### Phase 1 — Tensor Library

**Objective:** Create a minimal tensor abstraction and CPU execution backend.

**Tensor Structure:**
```c
typedef struct {
    float* data;
    int rows;
    int cols;
} Tensor;
```

**Operations:**
- Matrix Multiplication
- Element-wise Addition
- ReLU
- Softmax
- Transpose

**Deliverables:** Tensor creation API, memory management, CPU kernels, unit tests.

**Could do later:**
- Softmax over a user-specified axis (currently hardcoded to the last axis, which covers the common logits/attention case). Non-last axes need stride-based gather since the reduced elements aren't contiguous.

---

### Phase 2 — Computational Graph

**Objective:** Represent computations as a directed acyclic graph (DAG).

**Example:**
```
A ────╮
       MatMul ──▶ Add ──▶ ReLU ──▶ Output
B ────╯           ▲
                  │
                  C
```

**Node Structure:**
```c
typedef struct {
    OpType op;
    Node** inputs;
    Tensor* output;
} Node;
```

**Deliverables:** Graph construction API, graph traversal, topological sorting.

---

### Phase 3 — Graph Optimizer

**Objective:** Apply compiler-style optimizations.

**Optimization Passes:**

| Pass | Description | Example |
|------|-------------|---------|
| Constant Folding | Evaluate constant sub-expressions at compile time | `2 + 3` → `5` |
| Dead Node Elimination | Remove nodes whose outputs are never consumed | — |
| Operator Fusion | Merge sequential ops into a single kernel | `MatMul → Add → ReLU` → `FusedMatMulAddReLU` |
| Redundant Op Removal | Eliminate no-op patterns | `Transpose(Transpose(X))` → `X` |

**Deliverables:** Pass manager, optimization framework, benchmark comparisons.

---

### Phase 4 — PTX Backend

**Objective:** Generate PTX assembly directly.

**Example output:**
```ptx
ld.global.f32  %f1, [%rd1];
ld.global.f32  %f2, [%rd2];
mul.f32        %f3, %f1, %f2;
st.global.f32  [%rd3], %f3;
```

**Components:** PTX instruction emitter, register allocator, kernel templates.

**Deliverables:** PTX generation engine, PTX file output, validation tests.

---

### Phase 5 — Runtime System

**Objective:** Load and execute generated PTX.

**Execution flow:**
```
Generated PTX
      │
      ▼
CUDA Driver API
      │
      ▼
GPU Execution
```

**Components:**
- **Memory Manager** — device allocation, host-device transfers, buffer reuse
- **Kernel Launcher** — kernel dispatch and configuration
- **Error Handling** — compilation failures, runtime failures, memory tracking

**Deliverables:** PTX loading, execution engine, profiling support.

---

### Phase 6 — Automatic Differentiation

**Objective:** Implement reverse-mode autodiff.

**Flow:**
```
Forward Pass → Loss → Backward Pass
```

**Components:** Gradient storage, backpropagation engine, gradient accumulation.

**Deliverables:** Training support, gradient verification tests.

---

### Phase 7 — Advanced Optimizations

**Kernel Fusion:**
```
MatMul → Add → ReLU   ──▶   Single Fused Kernel
```

**Memory Optimization:** Buffer reuse, tensor lifetime analysis, memory planning.

**Scheduling:** Execution ordering, dependency analysis.

---

### Phase 8 — Multi-GPU Research

**Objective:** Explore distributed execution concepts.

**Tensor Parallelism:**
```
Tensor → Split Across GPUs
```

**Pipeline Parallelism:**
```
GPU0 → GPU1 → GPU2 → GPU3
```

**Collective Operations:** AllReduce, Broadcast, Gather.

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Language | C (primary), Modern C++ (optional) |
| GPU | CUDA Driver API, PTX |
| Build | CMake |
| Testing | GoogleTest + custom validation suite |
| Profiling | Nsight Systems, Nsight Compute |

---

## Repository Structure

```
MiniXLA/
│
├── include/
├── src/
│   ├── tensor/
│   ├── graph/
│   ├── optimizer/
│   ├── ptx/
│   ├── runtime/
│   └── autodiff/
│
├── tests/
├── benchmarks/
├── examples/
├── docs/
└── CMakeLists.txt
```

---

## Success Criteria

| Milestone | Criterion |
|-----------|-----------|
| 1 | CPU tensor operations functional |
| 2 | Graph construction and execution working |
| 3 | Optimization passes implemented |
| 4 | PTX generation operational |
| 5 | GPU execution through generated PTX |
| 6 | Autodiff and training support |
| 7 | Multi-GPU proof of concept |

---

## Long-Term Vision

A completed MiniXLA project provides practical, hands-on understanding of the engineering concepts underpinning modern systems including **XLA**, **TensorRT**, **TVM**, **DeepSpeed**, **Megatron-LM**, and custom large-scale AI training stacks — spanning compiler design, GPU architecture, PTX programming, runtime systems, and distributed AI infrastructure.