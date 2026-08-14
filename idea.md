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

### Phase 1: Tensor Library

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

### Phase 2: Computational Graph

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

### Phase 3: Graph Optimizer

**Objective:** Apply compiler-style optimizations.

**Optimization Passes:**

| Pass | Description | Example |
|------|-------------|---------|
| Constant Folding | Evaluate constant sub-expressions at compile time | `2 + 3` → `5` |
| Dead Node Elimination | Remove nodes whose outputs are never consumed | N/A |
| Operator Fusion | Merge sequential ops into a single kernel | `MatMul → Add → ReLU` → `FusedMatMulAddReLU` |
| Redundant Op Removal | Eliminate no-op patterns | `Transpose(Transpose(X))` → `X` |

**Deliverables:** Pass manager, optimization framework, benchmark comparisons.

---

### Phase 4: PTX Backend

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

### Phase 5: Runtime System

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
- **Memory Manager**: device allocation, host-device transfers, buffer reuse
- **Kernel Launcher**: kernel dispatch and configuration
- **Error Handling**: compilation failures, runtime failures, memory tracking

**Deliverables:** PTX loading, execution engine, profiling support.

---

### Phase 6: Automatic Differentiation

**Objective:** Implement reverse-mode autodiff.

**Flow:**
```
Forward Pass → Loss → Backward Pass
```

**Components:** Gradient storage, backpropagation engine, gradient accumulation.

**Deliverables:** Training support, gradient verification tests.

---

### Phase 7: Advanced Optimizations

**Kernel Fusion:**
```
MatMul → Add → ReLU   ──▶   Single Fused Kernel
```

**Memory Optimization:** Buffer reuse, tensor lifetime analysis, memory planning.

**Scheduling:** Execution ordering, dependency analysis.

---

### Phase 8: Multi-GPU Research

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

### Phase 9: Tensor Cores and the PyTorch Comparison (status, not a plan)

**Objective:** actually attempt to beat PyTorch, on both speed and memory,
having built enough of the compiler (Phases 1-7) to make the attempt honest.

**What happened**, full detail in
[`docs/research/tensorcore-and-fusion.md`](docs/research/tensorcore-and-fusion.md):
a real TF32 tensor-core kernel (`mma.sync`, PTX ISA fragment layout,
correctness-verified) landed additively alongside the existing CUDA-core
emitter, but tops out at 51.4% of cuBLAS — behind the CUDA-core kernel it
sits next to, not ahead of it. Separately, MiniXLA's fused single-launch
execution of `matmul → add → relu` was measured against PyTorch eager:
**a real, whole-size-range win on peak GPU memory** (no intermediate
tensors ever materialize), and **a narrow win on latency only at the
smallest size tested** (128×128, ~21% faster) — PyTorch eager wins at
every other size measured. Neither raw GEMM nor general-size latency beats
PyTorch; memory does.

**Could do later:** shared-memory blocking for the tensor-core kernel hit
a real bug scaling the block tile past 32×32 (documented in the research
doc, a fixable cooperative-load assumption, not a fragment-layout issue);
warp-tiling (siboehm's kernel 10) as a lower-risk CUDA-core-only lever
never attempted this session; profiling `gpu_execute` under repeated calls
to check whether a buffer pool is actually load-bearing before building
one speculatively.

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

Sketched at inception, and **not** what got built:

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

The real tree is flat, with `docs/` the only subdirectory. At ~3,900 lines
across 21 files this would be six directories holding two files each, and
the file names (`tensor.c`, `graph.c`, `optimizer.c`, …) already carry the
grouping the tree was going to express. CMake likewise didn't survive
contact: the CPU targets build with `gcc` and the GPU targets with `nvcc`
(MSVC host), which is two toolchains in one project, and there's no
object-file step to make incremental. `build.sh` covers it in 30 lines.
See `README.md`'s Layout section for what's actually there.

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

A completed MiniXLA project provides practical, hands-on understanding of the engineering concepts underpinning modern systems including **XLA**, **TensorRT**, **TVM**, **DeepSpeed**, **Megatron-LM**, and custom large-scale AI training stacks, spanning compiler design, GPU architecture, PTX programming, runtime systems, and distributed AI infrastructure.