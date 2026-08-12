// Graph optimizer: in-place rewrite passes run to a fixpoint.
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

typedef int (*Pass)(Node**);

// Run one pass; if it changed the graph, free every node that was reachable
// immediately before the pass ran but isn't anymore. Snapshotting right
// before each call (rather than once at the start of optimize) is what makes
// this catch nodes the optimizer itself created and later superseded — e.g.
// a const folded from a matmul that a later fold then absorbs into a bigger
// const. A single start-of-optimize snapshot would never see that node.
static int run_pass(Node** root, Pass pass) {
    Node** before;
    int n_before = graph_collect(*root, &before);

    int changed = pass(root);
    if (changed) {
        Node** live;
        int n_live = graph_collect(*root, &live);
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
    free(before);
    return changed;
}

Node* optimize(Node* root) {
    int changed;
    do {
        changed = 0;
        changed |= run_pass(&root, redundant_op_removal);
        changed |= run_pass(&root, constant_folding);
        changed |= run_pass(&root, operator_fusion);
    } while (changed);
    return root;
}
