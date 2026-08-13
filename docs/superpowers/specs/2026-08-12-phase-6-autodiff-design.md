# Phase 6: Reverse-Mode Autodiff Design

## Goal

Compute gradients of a scalar loss (`sum(output)`) with respect to every node
in the graph that feeds it, so a training loop can update `OP_INPUT` weight
leaves. Scope, per `idea.md`: "gradient storage, backpropagation engine,
gradient accumulation" with "gradient verification tests" as the correctness
bar: not a differentiable backward pass (no second-order gradients), and
not GPU execution of the backward pass.

## Why eager tensors, not a symbolic backward graph

XLA/JAX-style autodiff differentiates the *graph*, producing new nodes the
optimizer can then fuse. That's the more powerful design, but it requires
new op types purely to express backward math as graph nodes (a comparison/
mask op for ReLU's gradient, a reduction op for softmax's Jacobian-vector
product), none of which have any forward-pass use. Adding op types (and
their `eval_op`, PTX, and GPU-runtime cases) that exist solely to make
*gradients* differentiable, when nothing in this codebase's scope needs a
gradient of a gradient, is exactly the premature generalization the project
standards rule out.

Instead: `backward(Node* output)` walks the already-executed graph (every
node's `->output` Tensor is memoized by a prior `execute(output)` call, same
as constant folding reads memoized outputs) and computes gradient **Tensors**
directly, per node, using a handful of private per-op backward functions.
This is a complete, correct, testable reverse-mode AD; it just isn't
re-differentiable. If a later phase needs meta-gradients, build the
symbolic version then; it isn't needed now.

## API (`autodiff.c` / `autodiff.h`)

```c
typedef struct { const Node* node; Tensor* grad; } GradEntry;
typedef struct { GradEntry* entries; int count; } GradTape;

// Requires execute(output) to have already been called (every reachable
// node's ->output populated). Computes d(sum(output))/d(node) for every
// node reachable from output, via reverse-mode accumulation. Returns a tape
// the caller looks up with backward_grad and frees with backward_free.
GradTape backward(Node* output);

// Linear scan; NULL if `node` has no entry (unreachable from the backward
// walk's root, or op is unsupported, see "OP_FUSED" below).
Tensor* backward_grad(const GradTape* tape, const Node* node);

void backward_free(GradTape* tape);
```

## Algorithm

1. **Ordering.** Reverse-mode accumulation requires a node be finalized (all
   of its consumers' contributions summed) before it pushes gradient to its
   own inputs. `graph_collect`'s pre-order DFS doesn't guarantee that for a
   DAG: a shared node can be reached through one consumer before a second
   consumer is even visited. `backward` instead does its own **post-order**
   DFS (recurse into inputs, emit self after, the reverse of
   `graph_collect`'s order), then reverses that list. A node's postorder
   position is after all of its dependencies, so after reversal it's after
   all of its dependents (consumers), exactly the order backward
   propagation needs. Implemented as a private `topo_rec` in `autodiff.c`,
   reusing the `visited` scratch flag convention (reset before returning,
   same contract as `graph_collect`).

2. **Seed.** `output`'s gradient is `ones_like(output->output)`: this is
   what makes the computed quantity `d(sum(output))/d(*)` rather than
   requiring `output` to already be scalar.

3. **Propagate.** For each node `x` in the reversed postorder (skipping any
   with no accumulated gradient, happens only for nodes outside the actual
   dependency chain to `output`, which can't occur since the traversal
   started at `output`, so this is defensive, not load-bearing): dispatch on
   `x->op`, compute each input's gradient contribution from `x`'s own
   accumulated gradient and `x`'s already-memoized forward tensors, and
   accumulate (sum) into that input's tape entry.

## Per-op gradient rules

- **OP_MATMUL** (`y = A @ B`): `dA = dY @ Bᵀ`, `dB = Aᵀ @ dY`, via the
  existing `matmul`/`transpose`: batch dimensions fall out for free since
  `transpose` only swaps the trailing two axes, matching how batched matmul
  is defined here.
- **OP_ADD** (`y = a + b`, `b` possibly broadcasting a trailing suffix of
  `a`'s shape, see the Phase 7 broadcast-add doc): `da = dY`; `db` is `dY`
  summed over the repeated (broadcast) positions back down to `b`'s shape:
  the exact inverse of the broadcast. Reduces to a plain copy when `b`
  wasn't actually broadcast, so one code path covers both cases.
- **OP_RELU** (`y = relu(x)`): `dx = dY` where `x > 0`, else `0`. One pass
  over the data; no separate mask tensor.
- **OP_TRANSPOSE**: `dx = transpose(dY)`; transpose is its own adjoint for
  a two-axis swap.
- **OP_SOFTMAX** (`y = softmax(x)`, row-wise): per row, standard VJP
  `dx_j = y_j * (dY_j - Σ_k dY_k · y_k)`. Computed directly per row (dot
  product then a combine pass) rather than composed from generic tensor
  ops: same "hand-roll the fused loop" choice `eval_fused` already makes.
- **OP_INPUT**: leaf; nothing to propagate further. Its own accumulated
  gradient stays in the tape for the caller to read (this is the point:
  `backward_grad(tape, weight_node)` is how a trainer gets `dLoss/dWeight`).
- **OP_FUSED**: not handled. `optimize()` is the only thing that ever
  creates `OP_FUSED` nodes, and nothing in this design calls `optimize()`
  before `backward()`: the intended flow is *build → backward → optimize
  the augmented (forward + gradient) graph → execute*. If `OP_FUSED` is
  ever encountered anyway (graph optimized before differentiating, which is
  a usage error), `backward` logs to stderr and treats it as a dead end
  rather than crashing.

## New tensor op: `tensor_mul`

Elementwise multiply, exact-shape only (no broadcast, nothing in this
codebase needs a broadcasting multiply). Added to `tensor.c`/`tensor.h`
alongside `tensor_add`, and exposed as a graph op (`OP_MUL` / `g_mul`) since
elementwise multiply is a generically useful tensor primitive on its own,
not something invented only to serve autodiff's internals: a tensor
library with `+` but no `*` is missing an obvious piece of its surface.
`OP_MUL` participates in `eval_op`/`execute` and constant folding exactly
like `OP_ADD`; it is **not** added to the fusion pass's epilogue pattern
(nothing currently produces a matmul → mul chain worth fusing, and adding
an unused fusion pattern is speculative).

## Verification

Numerical gradient checking (finite differences) is the standard, rigorous
correctness proof for autodiff: compare each analytic gradient entry
against `(f(x+ε) - f(x-ε)) / 2ε` for a scalar loss, per element, within a
tolerance that accounts for fp32 precision (`1e-2` relative, matching the
tolerance already used for GPU-vs-CPU comparisons at similar scale). Covers:
matmul, add (including broadcast bias), relu, transpose, softmax
individually, and a full two-layer MLP chain (matmul → add → relu → matmul
→ add) to catch accumulation bugs that a single-op test can't.

## Out of scope

- GPU execution of the backward pass (CPU-only, matching the "eager tensor"
  design above).
- Second-order gradients / differentiating through `backward` itself.
- `OP_SOFTMAX`/`OP_TRANSPOSE` inside fused regions' epilogues (unchanged
  from Phase 3: fusion only ever covers matmul + `OP_ADD`/`OP_RELU`
  chains, so this doesn't interact with autodiff at all).
