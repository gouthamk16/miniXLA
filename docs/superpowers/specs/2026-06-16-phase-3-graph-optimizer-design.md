# Phase 3: Graph Optimizer Design

## Goal

Add a compiler-style optimizer over the existing computational-graph DAG
(`graph.c`/`graph.h`). It rewrites the graph before execution to remove
redundant work and fuse operations, mirroring how XLA transforms its IR.

Scope: four passes (redundant-op removal, constant folding, operator fusion,
dead-node elimination) driven by a pass manager that runs to a fixpoint.
CPU only. Machine-code generation for fused regions is deliberately deferred
to Phase 4 (PTX).

## Core model

The optimizer **mutates the existing `Node` DAG in place**: passes rewire
pointers rather than rebuilding the graph. This is faster than a functional
"return a new graph" model: it allocates only for the nodes a rewrite
introduces (a handful) instead of cloning the whole DAG on every pass.

```c
Node* optimize(Node* root);   // mutates and returns the (possibly new) root
```

The pass manager runs the three simplifying passes (redundant-op removal,
constant folding, operator fusion) repeatedly until a full sweep produces no
change (fixpoint), then runs dead-node elimination once to free orphans.

## Graph additions (`graph.h` / `graph.c`)

### Constant marking
```c
Node* const_node(Tensor* t);   // leaf with is_const = 1
Node* input_node(Tensor* t);   // leaf with is_const = 0 (unchanged signature)
```
New field on `Node`: `char is_const;`. Constant folding fires only on nodes
whose inputs all trace back to const leaves.

### Fused region
New op `OP_FUSED`. A fused node represents a `matmul` anchor followed by a
chain of element-wise ops applied to each output cell:

```c
typedef struct { OpType op; Node* operand; } EpStep;
// op      : OP_ADD or OP_RELU
// operand : 2nd input node for OP_ADD; NULL for OP_RELU
```
Added to `Node`: `EpStep* epilogue; int n_epilogue;`.

Invariants:
- `inputs[0]`, `inputs[1]` are the matmul operands `a`, `b`.
- Every binary epilogue `operand` node is **also** present in `inputs[]`, so
  the existing `collect` / `free_graph` / dependency traversal stays correct.
- Epilogue ops are element-wise only (`OP_ADD`, `OP_RELU`). `softmax`
  (row reduction) and `transpose` (data movement) are never fused.
- Each `OP_ADD` operand has the same shape as the matmul output `[M, N]`
  (bias add, element-wise, no broadcasting).

### Refactor for reuse
Pull the op→kernel dispatch currently inside `execute` into a function
exposed in `graph.h` (non-static) so the optimizer can reuse it:
```c
Tensor* eval_op(Node* node, Tensor** in);
```
Both `execute` and constant folding call it. `execute` gains an `OP_FUSED`
case that runs the region in a single loop per output cell:
for each `(batch, i, j)` compute the dot-product, walk the epilogue applying
each step to that value in a register, then store once. No intermediate
tensors are allocated.

### Shared rewrite helper
```c
static void replace(Node* root, Node* old, Node* repl);
```
Repoints every input pointer equal to `old` (and the root, handled by the
caller via the returned root) to `repl`. O(V) per call; fine at this scale.

## Passes

Each pass returns whether it changed the graph, so the manager can detect the
fixpoint.

1. **Redundant-op removal**: algebraic identities on the graph:
   - `transpose(transpose(x))` → `x`
   - `relu(relu(x))` → `relu(x)`
   Replace the outer node with the inner subgraph via `replace`.

2. **Constant folding**: for any node whose inputs are all const leaves, run
   `eval_op` immediately, wrap the resulting tensor in a fresh `const_node`
   (which owns the tensor), and `replace` the node with it. The new const is
   itself foldable, so multi-op constant chains collapse over successive
   fixpoint iterations.

3. **Operator fusion**: compute consumer counts from the root. Find a
   `matmul` node whose output is single-use and flows into a chain of
   single-use element-wise ops. Single-use is required: if an intermediate
   has another consumer, fusing it away would change what that consumer sees.
   Collapse the matmul + element-wise chain into one `OP_FUSED` node and
   `replace` the chain's top node with it.

4. **Dead-node elimination**: `collect` the node set reachable from the root
   *before* optimizing; after the fixpoint, `collect` the *live* set from the
   new root; free every node in `before` that is not in `live`. Orphans are
   exactly `before − live`, so no global node registry is required. (Nodes
   created by passes are always connected, hence always live.)

## Pass ordering

All simplifying passes take `Node** root`, since any of them can replace the
root itself (e.g. when the root *is* the outer `transpose` of a
`transpose(transpose(x))`).

```
loop:
    changed  = redundant_op_removal(&root)
    changed |= constant_folding(&root)
    changed |= operator_fusion(&root)
    if !changed: break
dead_node_elimination(root, before_set)
```

## File layout

- `optimizer.c` / `optimizer.h`: the four passes and the pass manager
  (`optimize`).
- `graph.c` / `graph.h`: `is_const`, `const_node`, `OP_FUSED`, the `EpStep`
  type, the `eval_op` refactor, and the `OP_FUSED` execution case.

This keeps optimization logic separate from graph construction.

## Verification

A test (extending `main` or a dedicated test file) that:

1. **Fusion correctness**: builds `relu(matmul(a, b) + c)`, runs `optimize`,
   asserts (a) the optimized result is numerically identical to the
   unoptimized `execute`, and (b) the optimized graph is a single `OP_FUSED`
   node over the three input leaves.
2. **Constant folding**: `matmul(const a, const b)` collapses to a single
   const leaf holding the precomputed product.
3. **Redundant-op removal**: `transpose(transpose(x))` reduces to `x`.
4. **No leaks**: `free_graph` on the optimized root, with input tensors still
   owned by the caller, runs clean (manual review / sanitizer if available).

## Out of scope (later phases)

- Code generation for fused regions → Phase 4 (PTX).
- Additional fusion patterns (multiple anchors, longer element-wise
  vocabularies), add once the single-anchor path proves out.
- Broadcasting in fused `ADD`.
