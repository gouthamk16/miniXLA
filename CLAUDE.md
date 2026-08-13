# MiniXLA: project standards

A lightweight ML compiler and runtime exploring graph compilation, optimization,
and PTX code generation on NVIDIA GPUs. See `README.md` for what's built and
`idea.md` for the long-term phase roadmap. Design docs for completed phases
live under `docs/superpowers/`.

## Code standards

- Simplest correct solution over the extensible one. A function earns its
  existence by being reused or by making the code clearer, not by
  anticipating future need.
- No comments that restate what the code does. Comment only the non-obvious:
  an invariant that isn't visible locally, a workaround, why a constant is
  what it is.
- Functions stay small: one meaningful thing each. If a function is doing
  two jobs, split it.
- Every allocation the rest of the file already checks (see `tensor.c`,
  `graph.c`'s `execute`/`fused_node`) gets checked the same way: an
  inconsistent unchecked `malloc` next to five checked ones is a bug, not a
  style choice. Scratch/growth buffers with no realistic failure mode at this
  scale (e.g. the small bookkeeping reallocs in `graph_collect`, `ptx.c`'s
  string buffer) are the deliberate exception; don't add checks there.
- Duplicated logic (3+ occurrences) gets extracted; 2 occurrences usually
  don't yet.
- No stray `TODO` comments. Either do it now or don't write it down.
- Validate at system boundaries (public API entry points like `matmul`,
  `create_tensor`, graph constructors taking caller-supplied shapes). Don't
  add validation for states the code's own invariants already rule out.

## Architecture invariants

- The graph (`graph.c`/`graph.h`) is a DAG built via `g_*`/`input_node`/
  `const_node`/`fused_node` constructors. Nothing in this codebase mutates a
  node's `inputs[]` except the optimizer's `replace` helper.
- The optimizer (`optimizer.c`) mutates the DAG in place and runs its passes
  to a fixpoint. Every pass returns "did I change anything" so the manager
  can detect the fixpoint. Each pass call is followed by an immediate
  before/after reachability diff (`run_pass`) that frees anything the pass
  just orphaned, including nodes the optimizer itself created in an earlier
  pass call. Don't reintroduce a single start-of-`optimize` snapshot; it
  misses nodes created and later superseded mid-optimization (see the
  `test_constant_folding_chain` regression test).
- `OP_FUSED` epilogue operands must also appear in `inputs[]`: that's what
  keeps `graph_collect`/`free_graph`/dependency traversal correct without a
  separate bookkeeping structure. Anywhere a node gets replaced (`optimizer.c`'s
  `replace`), both `inputs[]` *and* `epilogue[].operand` must be repointed:
  a real dangling-pointer bug shipped from only doing the first (see the
  `test_fusion_diamond_epilogue` regression test).
- Two traversal orders exist for a reason. Don't collapse them. `graph_collect`
  (preorder) is for plain reachability/collection. `graph_topo_order`
  (postorder: every node after its own inputs) is for anything that executes
  or accumulates in dependency order: `gpu_execute`'s kernel scheduling and
  `autodiff.c`'s backward pass (which walks it in reverse) both require this.
  Reversing `graph_collect`'s output is **not** a valid topological order for
  a DAG with a shared node, only for a tree/chain; that exact bug shipped
  once (see the `test_diamond_shared_input` / `test_deep_chain` regression
  tests).
- The PTX emitter (`ptx.c`) and both GPU runtimes (`runtime.c`, `gpu_exec.c`)
  are 2-D-only by design (no batch dimension in the kernel) and assume every
  `OP_ADD` epilogue operand is full-size (`M*N`): broadcast bias-add
  (`tensor_add`'s trailing-suffix case) is CPU-only; both GPU runtimes detect
  a broadcast-shaped operand and fail loudly rather than compute silently
  wrong output. The CPU path (`tensor.c`, `graph.c`) does support batched
  matmul; don't assume GPU and CPU code paths share either constraint.
- `autodiff.c`'s `backward()` is CPU-only and expects the *unoptimized*
  graph: call it before `optimize()`, not after. It computes gradient
  Tensors directly rather than building a second differentiable graph (see
  the Phase 6 design doc for why); don't add graph-node VJP rules expecting
  `backward` output to be itself differentiable.

## Build & test

See `README.md` for the exact commands. In short: `gcc` builds everything
that doesn't touch CUDA (`demo`, `tests`); `nvcc` builds anything that links
`gpu_exec.c`/`runtime.c` (`gpu_test`). Run both test binaries before
considering a change to `graph.c`, `optimizer.c`, `tensor.c`, `ptx.c`,
`runtime.c`, or `gpu_exec.c` done: the CPU and GPU paths execute the same
optimized graphs and are cross-checked against each other, so a change that
only passes one side is not verified.
