# Phase 3 — Graph Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compiler-style optimizer that rewrites the Node DAG in place — redundant-op removal, constant folding, operator fusion, dead-node elimination — driven by a pass manager that runs to a fixpoint.

**Architecture:** Passes mutate the existing graph in place and return "did I change anything?" so a manager loops them to a fixpoint, then frees orphaned nodes. Fusion collapses a single-use `matmul` + element-wise chain into one `OP_FUSED` node whose executor runs the whole region in one loop per output cell, allocating no intermediates. Optimizer logic lives in its own translation unit; graph-structure additions live in `graph.c`/`graph.h`.

**Tech Stack:** C (C99), gcc, `-lm`. Tests are `assert`-based in a standalone `tests.c`.

---

## File structure

- `tensor.c` / `tensor.h` — unchanged (kernels).
- `graph.h` / `graph.c` — graph construction + execution. Gains `is_const`, `uses`, `OP_FUSED`, `EpStep`, `const_node`, `fused_node`, public `eval_op`, public `graph_collect`. `main()` moves out.
- `main.c` — the demo `main()` extracted from `graph.c` (keeps the library linkable against a test harness).
- `optimizer.h` / `optimizer.c` — the four passes, the helpers (`replace`, `count_uses`, `sole_consumer`), and `optimize`.
- `tests.c` — `assert`-based test harness with its own `main()`.

Build commands used throughout:
- Demo: `gcc -O2 -Wall -o demo main.c graph.c tensor.c optimizer.c -lm`
- Tests: `gcc -O2 -Wall -o tests tests.c graph.c tensor.c optimizer.c -lm`

(Before Task 5 creates `optimizer.c`, drop `optimizer.c` from the test command.)

---

## Task 1: Extract demo `main()` into `main.c`

**Files:**
- Create: `main.c`
- Modify: `graph.c` (remove `main()` and `#include <stdio.h>` only if now unused — keep it; `fprintf` still used)

- [ ] **Step 1: Create `main.c` with the existing demo**

```c
#include <stdio.h>
#include <stdlib.h>
#include "graph.h"

int main(void) {
    Tensor* a = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2);
    Tensor* b = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){3, 2}, 2);
    Tensor* c = create_tensor((float[]){-100, 0, 0, -100}, (int[]){2, 2}, 2);

    Node* na = input_node(a);
    Node* nb = input_node(b);
    Node* nc = input_node(c);
    Node* n1 = g_matmul(na, nb);
    Node* n2 = g_add(n1, nc);
    Node* n3 = g_relu(n2);

    Tensor* result = execute(n3);
    if (!result) { fprintf(stderr, "graph execution failed\n"); return EXIT_FAILURE; }

    printf("relu(matmul(a,b) + c) (expect [[0,28],[49,0]]):\n");
    print_tensor(result);

    free_graph(n3);
    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
    return EXIT_SUCCESS;
}
```

- [ ] **Step 2: Remove `main()` from `graph.c`**

Delete the entire `int main(void) { ... }` block at the bottom of `graph.c`. Nothing else changes.

- [ ] **Step 3: Build and run the demo**

Run: `gcc -O2 -Wall -o demo main.c graph.c tensor.c -lm && ./demo`
Expected: prints `[[0, 28],` / `[49, 0]]` and exits 0.

- [ ] **Step 4: Commit**

```bash
git add main.c graph.c
git commit -m "refactor: extract graph demo main into main.c"
```

---

## Task 2: Public `graph_collect` (reset-safe traversal)

Expose the node-collection traversal so the optimizer can enumerate the graph repeatedly. It resets the `visited` flag before returning so it is reusable.

**Files:**
- Modify: `graph.h`, `graph.c`

- [ ] **Step 1: Declare in `graph.h`**

Add above `void free_graph(Node* root);`:
```c
// Collect every node reachable from `root` exactly once into *out (caller frees
// *out). Returns the count. Resets each node's visited flag before returning.
int graph_collect(Node* root, Node*** out);
```

- [ ] **Step 2: Implement in `graph.c`, reusing it from `free_graph`**

Replace the existing `collect` + `free_graph` block with:
```c
static void collect_rec(Node* n, Node*** out, int* count, int* cap) {
    if (!n || n->visited) return;
    n->visited = 1;
    if (*count == *cap) {
        *cap = *cap ? *cap * 2 : 8;
        *out = (Node**)realloc(*out, *cap * sizeof(Node*));
    }
    (*out)[(*count)++] = n;
    for (int i = 0; i < n->n_inputs; i++) collect_rec(n->inputs[i], out, count, cap);
}

int graph_collect(Node* root, Node*** out) {
    *out = NULL;
    int count = 0, cap = 0;
    collect_rec(root, out, &count, &cap);
    for (int i = 0; i < count; i++) (*out)[i]->visited = 0;
    return count;
}

void free_graph(Node* root) {
    Node** seen;
    int count = graph_collect(root, &seen);
    for (int i = 0; i < count; i++) {
        if (seen[i]->owns_output) free_tensor(seen[i]->output);
        free(seen[i]->inputs);
        free(seen[i]);
    }
    free(seen);
}
```

- [ ] **Step 3: Build the demo to confirm no regression**

Run: `gcc -O2 -Wall -o demo main.c graph.c tensor.c -lm && ./demo`
Expected: same `[[0, 28],` / `[49, 0]]`, exit 0, no warnings.

- [ ] **Step 4: Commit**

```bash
git add graph.h graph.c
git commit -m "refactor: expose graph_collect, share it with free_graph"
```

---

## Task 3: `is_const`, `const_node`, and public `eval_op`

**Files:**
- Modify: `graph.h`, `graph.c`
- Create: `tests.c`

- [ ] **Step 1: Write the failing test in `tests.c`**

```c
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "graph.h"

static int approx(const Tensor* t, const float* want) {
    for (size_t i = 0; i < t->size; i++)
        if (fabsf(t->data[i] - want[i]) > 1e-4f) return 0;
    return 1;
}

static void test_const_and_eval_op(void) {
    Tensor* a = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2);
    Tensor* b = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){3, 2}, 2);

    Node* ka = const_node(a);
    Node* kb = const_node(b);
    assert(ka->is_const == 1 && kb->is_const == 1);

    Node* in = input_node(a);
    assert(in->is_const == 0);

    Node* m = g_matmul(ka, kb);
    Tensor* got = eval_op(m, (Tensor*[]){a, b});
    assert(approx(got, (float[]){22, 28, 49, 64}));

    free_tensor(got);
    free(ka->inputs); free(ka);
    free(kb->inputs); free(kb);
    free(in->inputs); free(in);
    free(m->inputs); free(m);
    free_tensor(a);
    free_tensor(b);
    printf("test_const_and_eval_op PASS\n");
}

int main(void) {
    test_const_and_eval_op();
    printf("all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run it to confirm it fails to compile**

Run: `gcc -O2 -Wall -o tests tests.c graph.c tensor.c -lm`
Expected: FAIL — `const_node` and `eval_op` undeclared.

- [ ] **Step 3: Extend the `Node` struct and declarations in `graph.h`**

Replace the `OpType` enum and `Node` struct region with:
```c
typedef enum {
    OP_INPUT,
    OP_MATMUL,
    OP_ADD,
    OP_RELU,
    OP_SOFTMAX,
    OP_TRANSPOSE,
    OP_FUSED
} OpType;

struct Node;
typedef struct { OpType op; struct Node* operand; } EpStep;

typedef struct Node {
    OpType op;
    struct Node** inputs;
    int n_inputs;
    Tensor* output;
    int owns_output;
    char visited;     // scratch flag for graph traversals
    int uses;         // scratch consumer count (set by the optimizer)
    char is_const;    // 1 for const_node leaves
    EpStep* epilogue; // OP_FUSED only
    int n_epilogue;
} Node;
```

Add these declarations alongside the other constructors:
```c
Node* const_node(Tensor* t);
Node* fused_node(Node* a, Node* b, EpStep* epilogue, int n_epilogue);

// Run one node's op given its already-computed input tensors. Used by execute
// and by constant folding.
Tensor* eval_op(Node* node, Tensor** in);
```

- [ ] **Step 4: Implement `const_node` and `eval_op` in `graph.c`, route `execute` through `eval_op`**

Add `const_node` next to `input_node`:
```c
Node* const_node(Tensor* t) {
    Node* n = make_node(OP_INPUT, NULL, 0);
    if (n) { n->output = t; n->is_const = 1; }  // owns_output stays 0: caller owns t
    return n;
}
```

Add a fused executor and `eval_op` above `execute`:
```c
static Tensor* eval_fused(Node* node) {
    Tensor* a = node->inputs[0]->output;
    Tensor* b = node->inputs[1]->output;
    int M = a->shape[a->ndim - 2], K = a->shape[a->ndim - 1], N = b->shape[b->ndim - 1];
    size_t batch = a->size / ((size_t)M * K);

    int* os = (int*)malloc(a->ndim * sizeof(int));
    if (!os) return NULL;
    memcpy(os, a->shape, a->ndim * sizeof(int));
    os[a->ndim - 1] = N;
    Tensor* r = create_tensor(NULL, os, a->ndim);
    free(os);
    if (!r) return NULL;

    for (size_t bt = 0; bt < batch; bt++) {
        const float* A = a->data + bt * M * K;
        const float* B = b->data + bt * K * N;
        float* C = r->data + bt * M * N;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float x = 0;
                for (int k = 0; k < K; k++) x += A[i * K + k] * B[k * N + j];
                for (int e = 0; e < node->n_epilogue; e++) {
                    EpStep s = node->epilogue[e];
                    if (s.op == OP_ADD)       x += s.operand->output->data[bt * M * N + i * N + j];
                    else if (s.op == OP_RELU) x = x > 0 ? x : 0;
                }
                C[i * N + j] = x;
            }
        }
    }
    return r;
}

Tensor* eval_op(Node* node, Tensor** in) {
    switch (node->op) {
        case OP_MATMUL:    return matmul(in[0], in[1]);
        case OP_ADD:       return tensor_add(in[0], in[1]);
        case OP_RELU:      return relu(in[0]);
        case OP_SOFTMAX:   return softmax(in[0]);
        case OP_TRANSPOSE: return transpose(in[0]);
        case OP_FUSED:     return eval_fused(node);
        case OP_INPUT:     return NULL;
    }
    return NULL;
}
```

Replace the `switch` inside `execute` with a single call:
```c
    Tensor* out = eval_op(node, in);
    free(in);
```
(`#include <string.h>` is needed in `graph.c` for `memcpy` — add it if not present.)

- [ ] **Step 5: Build and run the test**

Run: `gcc -O2 -Wall -o tests tests.c graph.c tensor.c -lm && ./tests`
Expected: `test_const_and_eval_op PASS` then `all tests passed`.

- [ ] **Step 6: Confirm the demo still works**

Run: `gcc -O2 -Wall -o demo main.c graph.c tensor.c -lm && ./demo`
Expected: `[[0, 28],` / `[49, 0]]`.

- [ ] **Step 7: Commit**

```bash
git add graph.h graph.c tests.c
git commit -m "feat: add is_const/const_node and public eval_op with OP_FUSED exec"
```

---

## Task 4: `fused_node` constructor + execute a hand-built fused region

**Files:**
- Modify: `graph.c`, `tests.c`

- [ ] **Step 1: Add the failing test to `tests.c`**

Add this function and call it from `main` (before `all tests passed`):
```c
static void test_fused_exec(void) {
    Tensor* a = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2);
    Tensor* b = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){3, 2}, 2);
    Tensor* c = create_tensor((float[]){-100, 0, 0, -100}, (int[]){2, 2}, 2);

    Node* na = input_node(a), * nb = input_node(b), * nc = input_node(c);
    EpStep ep[2] = { { OP_ADD, nc }, { OP_RELU, NULL } };
    Node* f = fused_node(na, nb, ep, 2);

    Tensor* got = execute(f);
    assert(approx(got, (float[]){0, 28, 49, 0}));

    free_graph(f);
    free_tensor(a); free_tensor(b); free_tensor(c);
    printf("test_fused_exec PASS\n");
}
```

- [ ] **Step 2: Run to confirm failure**

Run: `gcc -O2 -Wall -o tests tests.c graph.c tensor.c -lm`
Expected: FAIL — `fused_node` undeclared (declared in Task 3's header but not yet defined).

- [ ] **Step 3: Implement `fused_node` in `graph.c`**

Add next to the other `g_*` constructors:
```c
Node* fused_node(Node* a, Node* b, EpStep* epilogue, int n_epilogue) {
    int n_operands = 0;
    for (int i = 0; i < n_epilogue; i++) if (epilogue[i].op == OP_ADD) n_operands++;

    int n_in = 2 + n_operands;
    Node** ins = (Node**)malloc(n_in * sizeof(Node*));
    if (!ins) return NULL;
    ins[0] = a; ins[1] = b;
    int k = 2;
    for (int i = 0; i < n_epilogue; i++) if (epilogue[i].op == OP_ADD) ins[k++] = epilogue[i].operand;

    Node* n = make_node(OP_FUSED, ins, n_in);
    free(ins);
    if (!n) return NULL;

    n->epilogue = (EpStep*)malloc(n_epilogue * sizeof(EpStep));
    if (!n->epilogue) { free(n->inputs); free(n); return NULL; }
    memcpy(n->epilogue, epilogue, n_epilogue * sizeof(EpStep));
    n->n_epilogue = n_epilogue;
    return n;
}
```

- [ ] **Step 4: Free the epilogue in `free_graph`**

In `free_graph`'s loop, add `free(seen[i]->epilogue);` before `free(seen[i]);` (it is `NULL` for non-fused nodes, so `free(NULL)` is a safe no-op):
```c
    for (int i = 0; i < count; i++) {
        if (seen[i]->owns_output) free_tensor(seen[i]->output);
        free(seen[i]->inputs);
        free(seen[i]->epilogue);
        free(seen[i]);
    }
```

- [ ] **Step 5: Build and run**

Run: `gcc -O2 -Wall -o tests tests.c graph.c tensor.c -lm && ./tests`
Expected: `test_fused_exec PASS`.

- [ ] **Step 6: Commit**

```bash
git add graph.c tests.c
git commit -m "feat: fused_node constructor and free_graph epilogue cleanup"
```

---

## Task 5: Optimizer skeleton + redundant-op removal

**Files:**
- Create: `optimizer.h`, `optimizer.c`
- Modify: `tests.c`

- [ ] **Step 1: Add the failing test to `tests.c`**

```c
#include "optimizer.h"   // add near the top with the other includes

static void test_redundant_removal(void) {
    Tensor* x = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2);
    Node* nx = input_node(x);
    Node* t1 = g_transpose(nx);
    Node* t2 = g_transpose(t1);   // transpose(transpose(x)) == x

    Node* root = optimize(t2);
    assert(root == nx);           // collapsed back to the original leaf

    Tensor* got = execute(root);
    assert(approx(got, (float[]){1, 2, 3, 4, 5, 6}));

    free_graph(root);
    free_tensor(x);
    printf("test_redundant_removal PASS\n");
}
```
Call `test_redundant_removal();` from `main`.

- [ ] **Step 2: Run to confirm failure**

Run: `gcc -O2 -Wall -o tests tests.c graph.c tensor.c -lm`
Expected: FAIL — `optimizer.h` not found / `optimize` undeclared.

- [ ] **Step 3: Create `optimizer.h`**

```c
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "graph.h"

// Optimize the graph rooted at `root` in place. Runs the simplifying passes to
// a fixpoint, then frees orphaned nodes. Returns the (possibly new) root.
Node* optimize(Node* root);

#endif
```

- [ ] **Step 4: Create `optimizer.c` with the shared helper, redundant-op removal, and a manager that runs only this pass for now**

```c
#include <stdlib.h>
#include "optimizer.h"

// Repoint every input pointer equal to `old` to `repl` across the graph.
static void replace(Node* root, Node* old, Node* repl) {
    Node** nodes;
    int n = graph_collect(root, &nodes);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < nodes[i]->n_inputs; j++)
            if (nodes[i]->inputs[j] == old) nodes[i]->inputs[j] = repl;
    free(nodes);
}

static int redundant_op_removal(Node** root) {
    Node** nodes;
    int n = graph_collect(*root, &nodes);
    int changed = 0;
    for (int i = 0; i < n; i++) {
        Node* x = nodes[i];
        Node* inner = NULL;
        if (x->op == OP_TRANSPOSE && x->inputs[0]->op == OP_TRANSPOSE)
            inner = x->inputs[0]->inputs[0];          // transpose(transpose(y)) -> y
        else if (x->op == OP_RELU && x->inputs[0]->op == OP_RELU)
            inner = x->inputs[0];                     // relu(relu(y)) -> relu(y)
        if (inner) {
            replace(*root, x, inner);
            if (*root == x) *root = inner;
            changed = 1;
            break;
        }
    }
    free(nodes);
    return changed;
}

Node* optimize(Node* root) {
    int changed;
    do {
        changed = 0;
        changed |= redundant_op_removal(&root);
    } while (changed);
    return root;
}
```

(Dead-node elimination is added in Task 8; until then the orphaned outer transpose leaks. That is acceptable mid-plan and is fixed before completion.)

- [ ] **Step 5: Build and run**

Run: `gcc -O2 -Wall -o tests tests.c graph.c tensor.c optimizer.c -lm && ./tests`
Expected: `test_redundant_removal PASS`.

- [ ] **Step 6: Commit**

```bash
git add optimizer.h optimizer.c tests.c
git commit -m "feat: optimizer skeleton with redundant-op removal pass"
```

---

## Task 6: Constant folding

**Files:**
- Modify: `optimizer.c`, `tests.c`

- [ ] **Step 1: Add the failing test to `tests.c`**

```c
static void test_constant_folding(void) {
    Tensor* a = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2);
    Tensor* b = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){3, 2}, 2);
    Node* ka = const_node(a);
    Node* kb = const_node(b);
    Node* m = g_matmul(ka, kb);

    Node* root = optimize(m);
    assert(root->op == OP_INPUT && root->is_const == 1);   // folded to one const leaf
    assert(approx(root->output, (float[]){22, 28, 49, 64}));

    free_graph(root);
    free_tensor(a);
    free_tensor(b);
    printf("test_constant_folding PASS\n");
}
```
Call `test_constant_folding();` from `main`.

- [ ] **Step 2: Run to confirm failure**

Run: `gcc -O2 -Wall -o tests tests.c graph.c tensor.c optimizer.c -lm && ./tests`
Expected: FAIL — `root->op` is `OP_MATMUL`, assertion fails (folding not implemented yet).

- [ ] **Step 3: Add the constant-folding pass to `optimizer.c`**

Add above `optimize`:
```c
static int constant_folding(Node** root) {
    Node** nodes;
    int n = graph_collect(*root, &nodes);
    int changed = 0;
    for (int i = 0; i < n; i++) {
        Node* x = nodes[i];
        if (x->op == OP_INPUT || x->op == OP_FUSED || x->n_inputs == 0) continue;

        int all_const = 1;
        for (int j = 0; j < x->n_inputs; j++)
            if (!x->inputs[j]->is_const) { all_const = 0; break; }
        if (!all_const) continue;

        Tensor** in = (Tensor**)malloc(x->n_inputs * sizeof(Tensor*));
        for (int j = 0; j < x->n_inputs; j++) in[j] = x->inputs[j]->output;
        Tensor* t = eval_op(x, in);
        free(in);
        if (!t) continue;

        Node* k = const_node(t);
        k->owns_output = 1;            // this tensor was produced by folding
        replace(*root, x, k);
        if (*root == x) *root = k;
        changed = 1;
        break;
    }
    free(nodes);
    return changed;
}
```

Wire it into the manager:
```c
Node* optimize(Node* root) {
    int changed;
    do {
        changed = 0;
        changed |= redundant_op_removal(&root);
        changed |= constant_folding(&root);
    } while (changed);
    return root;
}
```

- [ ] **Step 4: Build and run**

Run: `gcc -O2 -Wall -o tests tests.c graph.c tensor.c optimizer.c -lm && ./tests`
Expected: `test_constant_folding PASS` (and all prior tests still pass).

- [ ] **Step 5: Commit**

```bash
git add optimizer.c tests.c
git commit -m "feat: constant folding pass"
```

---

## Task 7: Operator fusion

**Files:**
- Modify: `optimizer.c`, `tests.c`

- [ ] **Step 1: Add the failing test to `tests.c`**

```c
static int count_nodes(Node* root) {
    Node** nodes;
    int n = graph_collect(root, &nodes);
    free(nodes);
    return n;
}

static void test_fusion(void) {
    Tensor* a = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2);
    Tensor* b = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){3, 2}, 2);
    Tensor* c = create_tensor((float[]){-100, 0, 0, -100}, (int[]){2, 2}, 2);

    Node* na = input_node(a), * nb = input_node(b), * nc = input_node(c);
    Node* root = g_relu(g_add(g_matmul(na, nb), nc));

    root = optimize(root);
    assert(root->op == OP_FUSED);
    assert(count_nodes(root) == 4);   // fused + a + b + c

    Tensor* got = execute(root);
    assert(approx(got, (float[]){0, 28, 49, 0}));

    free_graph(root);
    free_tensor(a); free_tensor(b); free_tensor(c);
    printf("test_fusion PASS\n");
}
```
Call `test_fusion();` from `main`.

- [ ] **Step 2: Run to confirm failure**

Run: `gcc -O2 -Wall -o tests tests.c graph.c tensor.c optimizer.c -lm && ./tests`
Expected: FAIL — `root->op` is `OP_RELU`, not `OP_FUSED`.

- [ ] **Step 3: Add use-counting, sole-consumer lookup, and the fusion pass to `optimizer.c`**

Add above `optimize`:
```c
static void count_uses(Node** nodes, int n) {
    for (int i = 0; i < n; i++) nodes[i]->uses = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < nodes[i]->n_inputs; j++)
            nodes[i]->inputs[j]->uses++;
}

// The unique consumer of `target`, or NULL if it has zero or more than one.
static Node* sole_consumer(Node** nodes, int n, Node* target) {
    Node* c = NULL;
    int cnt = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < nodes[i]->n_inputs; j++)
            if (nodes[i]->inputs[j] == target) { c = nodes[i]; cnt++; }
    return cnt == 1 ? c : NULL;
}

static int operator_fusion(Node** root) {
    Node** nodes;
    int n = graph_collect(*root, &nodes);
    count_uses(nodes, n);
    int changed = 0;

    for (int i = 0; i < n && !changed; i++) {
        if (nodes[i]->op != OP_MATMUL) continue;
        Node* m = nodes[i];

        EpStep ep[16];
        int n_ep = 0;
        Node* top = m;
        while (n_ep < 16) {
            Node* c = sole_consumer(nodes, n, top);   // single-use intermediate only
            if (!c) break;
            if (c->op == OP_ADD) {
                ep[n_ep].op = OP_ADD;
                ep[n_ep].operand = (c->inputs[0] == top) ? c->inputs[1] : c->inputs[0];
            } else if (c->op == OP_RELU) {
                ep[n_ep].op = OP_RELU;
                ep[n_ep].operand = NULL;
            } else {
                break;
            }
            n_ep++;
            top = c;
        }

        if (n_ep > 0) {
            Node* f = fused_node(m->inputs[0], m->inputs[1], ep, n_ep);
            replace(*root, top, f);
            if (*root == top) *root = f;
            changed = 1;
        }
    }
    free(nodes);
    return changed;
}
```

Wire it into the manager (fusion last, so folding/simplification happen first):
```c
Node* optimize(Node* root) {
    int changed;
    do {
        changed = 0;
        changed |= redundant_op_removal(&root);
        changed |= constant_folding(&root);
        changed |= operator_fusion(&root);
    } while (changed);
    return root;
}
```

- [ ] **Step 4: Build and run**

Run: `gcc -O2 -Wall -o tests tests.c graph.c tensor.c optimizer.c -lm && ./tests`
Expected: `test_fusion PASS` and all prior tests pass.

- [ ] **Step 5: Commit**

```bash
git add optimizer.c tests.c
git commit -m "feat: operator fusion pass (matmul + elementwise chain -> OP_FUSED)"
```

---

## Task 8: Dead-node elimination + leak-free fixpoint

Free the nodes orphaned by rewrites (the outer transpose, the pre-fusion matmul/add/relu, the folded sub-nodes).

**Files:**
- Modify: `optimizer.c`, `tests.c`

- [ ] **Step 1: Strengthen tests for orphan freeing**

Replace `test_redundant_removal`'s body assertion section to also confirm the old outer node is gone from the live set, and add a fused-orphan check. Append this test and call it from `main`:
```c
static void test_dce_after_fusion(void) {
    Tensor* a = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2);
    Tensor* b = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){3, 2}, 2);
    Tensor* c = create_tensor((float[]){-100, 0, 0, -100}, (int[]){2, 2}, 2);

    Node* na = input_node(a), * nb = input_node(b), * nc = input_node(c);
    Node* root = optimize(g_relu(g_add(g_matmul(na, nb), nc)));

    // Only the fused node plus the three leaves remain reachable.
    assert(count_nodes(root) == 4);
    assert(root->op == OP_FUSED);

    free_graph(root);
    free_tensor(a); free_tensor(b); free_tensor(c);
    printf("test_dce_after_fusion PASS\n");
}
```

- [ ] **Step 2: Run under a leak sanitizer to confirm the current leak**

Run: `gcc -O1 -g -fsanitize=address -o tests_asan tests.c graph.c tensor.c optimizer.c -lm && ./tests_asan`
Expected: tests pass but AddressSanitizer reports leaked `Node` allocations (the orphaned matmul/add/relu). If ASan is unavailable on the platform, skip this step and rely on Step 4's reasoning.

- [ ] **Step 3: Add dead-node elimination and capture the pre-optimization set in `optimize`**

Add above `optimize`:
```c
static void dead_node_elimination(Node* root, Node** before, int n_before) {
    Node** live;
    int n_live = graph_collect(root, &live);
    for (int i = 0; i < n_before; i++) {
        Node* b = before[i];
        int alive = 0;
        for (int j = 0; j < n_live; j++) if (live[j] == b) { alive = 1; break; }
        if (!alive) {
            if (b->owns_output) free_tensor(b->output);
            free(b->inputs);
            free(b->epilogue);
            free(b);
        }
    }
    free(live);
}
```

Update `optimize` to snapshot the original nodes and run DCE once at the end:
```c
Node* optimize(Node* root) {
    Node** before;
    int n_before = graph_collect(root, &before);

    int changed;
    do {
        changed = 0;
        changed |= redundant_op_removal(&root);
        changed |= constant_folding(&root);
        changed |= operator_fusion(&root);
    } while (changed);

    dead_node_elimination(root, before, n_before);
    free(before);
    return root;
}
```

- [ ] **Step 4: Rebuild and rerun under the sanitizer**

Run: `gcc -O1 -g -fsanitize=address -o tests_asan tests.c graph.c tensor.c optimizer.c -lm && ./tests_asan`
Expected: all tests pass, **no leaks reported**. (Caller still owns the input tensors `a`/`b`/`c`; DCE frees only orphaned nodes, and only their tensors when `owns_output` is set — which for orphaned matmul/add/relu it is not, because their outputs were never executed.)

- [ ] **Step 5: Build and run the normal test binary**

Run: `gcc -O2 -Wall -o tests tests.c graph.c tensor.c optimizer.c -lm && ./tests`
Expected: every test prints PASS, ends with `all tests passed`.

- [ ] **Step 6: Commit**

```bash
git add optimizer.c tests.c
git commit -m "feat: dead-node elimination, leak-free optimize fixpoint"
```

---

## Task 9: Demo the optimizer end-to-end

Show optimize in the demo so the phase has a visible artifact, and confirm parity with the unoptimized path.

**Files:**
- Modify: `main.c`

- [ ] **Step 1: Update `main.c` to run both paths and compare**

Replace `main.c` body with:
```c
#include <stdio.h>
#include <stdlib.h>
#include "graph.h"
#include "optimizer.h"

int main(void) {
    Tensor* a = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2);
    Tensor* b = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){3, 2}, 2);
    Tensor* c = create_tensor((float[]){-100, 0, 0, -100}, (int[]){2, 2}, 2);

    Node* root = g_relu(g_add(g_matmul(input_node(a), input_node(b)), input_node(c)));
    root = optimize(root);

    printf("optimized root op = %d (OP_FUSED expected)\n", root->op);
    printf("relu(matmul(a,b) + c) (expect [[0,28],[49,0]]):\n");
    print_tensor(execute(root));

    free_graph(root);
    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
    return EXIT_SUCCESS;
}
```

- [ ] **Step 2: Build and run**

Run: `gcc -O2 -Wall -o demo main.c graph.c tensor.c optimizer.c -lm && ./demo`
Expected: `optimized root op = 6 (OP_FUSED expected)` then `[[0, 28],` / `[49, 0]]`.

- [ ] **Step 3: Commit**

```bash
git add main.c
git commit -m "demo: run the optimizer end-to-end in main"
```

---

## Self-review notes

- **Spec coverage:** in-place mutation (Task 5–7), `const_node`/`is_const` (Task 3), `OP_FUSED`+`EpStep`+single-loop exec (Task 3–4), `eval_op` refactor (Task 3), `replace` helper (Task 5), redundant-op removal (Task 5), constant folding (Task 6), operator fusion with single-use guard (Task 7), dead-node elimination via `before − live` (Task 8), all three verification cases — fusion correctness, constant folding, transpose-of-transpose — plus a no-leak check (Tasks 5–8). File layout (`optimizer.*` vs `graph.*`) matches the spec.
- **Deviation from spec:** Task 1 (extract `main` into `main.c`) is an added prerequisite the spec implied — it is required so `tests.c` can own `main()`. The `eval_op` signature was already corrected to non-static in the spec; this plan matches it.
- **Type consistency:** `EpStep { OpType op; Node* operand; }`, `optimize(Node*)`, `eval_op(Node*, Tensor**)`, `graph_collect(Node*, Node***)`, `const_node`/`fused_node` signatures are used identically across all tasks.
