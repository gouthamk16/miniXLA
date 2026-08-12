#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "graph.h"
#include "optimizer.h"
#include "ptx.h"
#include "autodiff.h"

static int approx(const Tensor* t, const float* want) {
    for (size_t i = 0; i < t->size; i++)
        if (fabsf(t->data[i] - want[i]) > 1e-4f) return 0;
    return 1;
}

static void test_broadcast_add(void) {
    Tensor* a = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2);
    Tensor* bias = create_tensor((float[]){10, 20, 30}, (int[]){3}, 1);
    Tensor* got = tensor_add(a, bias);
    assert(approx(got, (float[]){11, 22, 33, 14, 25, 36}));
    free_tensor(got);

    // Exact-shape add still takes the same code path correctly.
    Tensor* b = create_tensor((float[]){1, 1, 1, 1, 1, 1}, (int[]){2, 3}, 2);
    Tensor* got2 = tensor_add(a, b);
    assert(approx(got2, (float[]){2, 3, 4, 5, 6, 7}));
    free_tensor(got2);

    // Mismatched, non-suffix shape is rejected, not silently misread.
    Tensor* bad = create_tensor((float[]){1, 2}, (int[]){2}, 1);
    assert(tensor_add(a, bad) == NULL);

    free_tensor(a); free_tensor(bias); free_tensor(b); free_tensor(bad);
    printf("test_broadcast_add PASS\n");
}

static void test_mul(void) {
    Tensor* a = create_tensor((float[]){1, 2, 3, 4}, (int[]){2, 2}, 2);
    Tensor* b = create_tensor((float[]){5, 6, 7, 8}, (int[]){2, 2}, 2);
    Tensor* got = tensor_mul(a, b);
    assert(approx(got, (float[]){5, 12, 21, 32}));
    free_tensor(got);
    free_tensor(a); free_tensor(b);
    printf("test_mul PASS\n");
}

static void test_fused_broadcast_bias(void) {
    // relu(matmul(a,b) + bias[N]) -- bias broadcasts across matmul's rows.
    // a[2,3] @ b[3,2] = [[22,28],[49,64]]; + bias[-30,-60] = [[-8,-32],[19,4]]
    Tensor* a = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2);
    Tensor* b = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){3, 2}, 2);
    Tensor* bias = create_tensor((float[]){-30, -60}, (int[]){2}, 1);

    Node* root = g_relu(g_add(g_matmul(input_node(a), input_node(b)), input_node(bias)));
    root = optimize(root);
    assert(root->op == OP_FUSED);

    Tensor* got = execute(root);
    assert(approx(got, (float[]){0, 0, 19, 4}));

    free_graph(root);
    free_tensor(a); free_tensor(b); free_tensor(bias);
    printf("test_fused_broadcast_bias PASS\n");
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

static void test_constant_folding_chain(void) {
    // Two-level constant chain: add(matmul(const_a, const_b), const_c). The
    // matmul folds first, then the add folds the result with const_c. Also
    // exercises that the optimizer frees the intermediate const the first
    // fold produces once the second fold subsumes it.
    Tensor* a = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2);
    Tensor* b = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){3, 2}, 2);
    Tensor* c = create_tensor((float[]){1, 1, 1, 1}, (int[]){2, 2}, 2);
    Node* m = g_matmul(const_node(a), const_node(b));
    Node* root = optimize(g_add(m, const_node(c)));

    assert(root->op == OP_INPUT && root->is_const == 1);
    assert(approx(root->output, (float[]){23, 29, 50, 65}));

    free_graph(root);
    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
    printf("test_constant_folding_chain PASS\n");
}

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

static void test_ptx_codegen(void) {
    Tensor* a = create_tensor((float[]){1,2,3,4,5,6}, (int[]){2,3}, 2);
    Tensor* b = create_tensor((float[]){1,2,3,4,5,6}, (int[]){3,2}, 2);
    Tensor* c = create_tensor((float[]){-100,0,0,-100}, (int[]){2,2}, 2);

    Node* root = optimize(g_relu(g_add(g_matmul(input_node(a), input_node(b)), input_node(c))));
    assert(root->op == OP_FUSED);

    char* ptx = emit_ptx(root);
    assert(ptx);
    assert(strstr(ptx, ".entry fused"));
    assert(strstr(ptx, "fma.rn.f32"));
    assert(strstr(ptx, "max.f32"));

    FILE* f = fopen("_test.ptx", "w");
    assert(f);
    fputs(ptx, f);
    fclose(f);

    int r = system("ptxas -arch=sm_89 _test.ptx -o _test.cubin");
    remove("_test.ptx");
    remove("_test.cubin");
    assert(r == 0);

    free(ptx);
    free_graph(root);
    free_tensor(a); free_tensor(b); free_tensor(c);
    printf("test_ptx_codegen PASS\n");
}

// ---- Gradient checking: compare backward()'s analytic gradient against a
// central-difference numerical one. execute() memoizes every node's output,
// so re-evaluating after perturbing a leaf requires clearing the memoized
// (non-leaf) outputs first or execute() just returns the stale cached
// result.

static void clear_outputs(Node* root) {
    Node** nodes;
    int n = graph_collect(root, &nodes);
    for (int i = 0; i < n; i++) {
        Node* nd = nodes[i];
        if (nd->n_inputs > 0) {   // leaves' output is caller-owned data, not memoized
            if (nd->owns_output) free_tensor(nd->output);
            nd->output = NULL;
            nd->owns_output = 0;
        }
    }
    free(nodes);
}

static float sum_tensor(const Tensor* t) {
    float s = 0;
    for (size_t i = 0; i < t->size; i++) s += t->data[i];
    return s;
}

// Perturbs leaf->output's data in place (restored after each element).
static int grad_check(Node* root, Node* leaf, GradTape* tape, float eps, float tol) {
    Tensor* analytic = backward_grad(tape, leaf);
    if (!analytic) { fprintf(stderr, "grad_check: no gradient entry for leaf\n"); return 0; }

    Tensor* x = leaf->output;
    for (size_t i = 0; i < x->size; i++) {
        float orig = x->data[i];

        x->data[i] = orig + eps;
        clear_outputs(root);
        float plus = sum_tensor(execute(root));

        x->data[i] = orig - eps;
        clear_outputs(root);
        float minus = sum_tensor(execute(root));

        x->data[i] = orig;
        clear_outputs(root);
        execute(root);

        float numeric = (plus - minus) / (2 * eps);
        float diff = fabsf(numeric - analytic->data[i]);
        float scale = fmaxf(1.0f, fabsf(numeric));
        if (diff / scale > tol) {
            fprintf(stderr, "grad_check: mismatch at %lu: analytic=%g numeric=%g\n",
                    (unsigned long)i, (double)analytic->data[i], (double)numeric);
            return 0;
        }
    }
    return 1;
}

static void test_grad_matmul(void) {
    Tensor* a = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2);
    Tensor* b = create_tensor((float[]){0.5f, -1, 2, 1, -0.5f, 1.5f}, (int[]){3, 2}, 2);
    Node* na = input_node(a), * nb = input_node(b);
    Node* root = g_matmul(na, nb);
    execute(root);

    GradTape tape = backward(root);
    assert(grad_check(root, na, &tape, 1e-3f, 1e-2f));
    assert(grad_check(root, nb, &tape, 1e-3f, 1e-2f));

    backward_free(&tape);
    free_graph(root);
    free_tensor(a); free_tensor(b);
    printf("test_grad_matmul PASS\n");
}

static void test_grad_broadcast_add(void) {
    Tensor* a = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2);
    Tensor* bias = create_tensor((float[]){10, -20, 30}, (int[]){3}, 1);
    Node* na = input_node(a), * nbias = input_node(bias);
    Node* root = g_add(na, nbias);
    execute(root);

    GradTape tape = backward(root);
    assert(grad_check(root, na, &tape, 1e-3f, 1e-2f));
    assert(grad_check(root, nbias, &tape, 1e-3f, 1e-2f));

    backward_free(&tape);
    free_graph(root);
    free_tensor(a); free_tensor(bias);
    printf("test_grad_broadcast_add PASS\n");
}

static void test_grad_relu(void) {
    // Kept away from 0 -- relu is non-differentiable there, which would
    // make finite-difference checking flaky rather than wrong.
    Tensor* x = create_tensor((float[]){-2, -1, 3, 4, -0.5f, 2}, (int[]){2, 3}, 2);
    Node* nx = input_node(x);
    Node* root = g_relu(nx);
    execute(root);

    GradTape tape = backward(root);
    assert(grad_check(root, nx, &tape, 1e-3f, 1e-2f));

    backward_free(&tape);
    free_graph(root);
    free_tensor(x);
    printf("test_grad_relu PASS\n");
}

static void test_grad_transpose(void) {
    Tensor* x = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2);
    Node* nx = input_node(x);
    Node* root = g_transpose(nx);
    execute(root);

    GradTape tape = backward(root);
    assert(grad_check(root, nx, &tape, 1e-3f, 1e-2f));

    backward_free(&tape);
    free_graph(root);
    free_tensor(x);
    printf("test_grad_transpose PASS\n");
}

static void test_grad_softmax(void) {
    Tensor* x = create_tensor((float[]){1, 2, 3, 0.5f, -1, 2}, (int[]){2, 3}, 2);
    Node* nx = input_node(x);
    Node* root = g_softmax(nx);
    execute(root);

    GradTape tape = backward(root);
    assert(grad_check(root, nx, &tape, 1e-3f, 1e-2f));

    backward_free(&tape);
    free_graph(root);
    free_tensor(x);
    printf("test_grad_softmax PASS\n");
}

static void test_grad_mlp(void) {
    // hidden = relu(x @ W1 + b1) -- the real MLP shape autodiff exists for.
    Tensor* x  = create_tensor((float[]){1, -2, 3, 0.5f}, (int[]){1, 4}, 2);
    Tensor* w1 = create_tensor((float[]){0.1f,-0.2f,0.3f,0.05f,-0.1f,0.2f,
                                          0.15f,-0.05f,0.25f,0.1f,-0.3f,0.2f},
                                (int[]){4, 3}, 2);
    Tensor* b1 = create_tensor((float[]){0.1f, -0.2f, 0.05f}, (int[]){3}, 1);

    Node* nx = input_node(x), * nw1 = input_node(w1), * nb1 = input_node(b1);
    Node* root = g_relu(g_add(g_matmul(nx, nw1), nb1));
    execute(root);

    GradTape tape = backward(root);
    assert(grad_check(root, nx,  &tape, 1e-3f, 1e-2f));
    assert(grad_check(root, nw1, &tape, 1e-3f, 1e-2f));
    assert(grad_check(root, nb1, &tape, 1e-3f, 1e-2f));

    backward_free(&tape);
    free_graph(root);
    free_tensor(x); free_tensor(w1); free_tensor(b1);
    printf("test_grad_mlp PASS\n");
}

int main(void) {
    test_broadcast_add();
    test_mul();
    test_fused_broadcast_bias();
    test_const_and_eval_op();
    test_fused_exec();
    test_redundant_removal();
    test_constant_folding();
    test_constant_folding_chain();
    test_fusion();
    test_dce_after_fusion();
    test_ptx_codegen();
    test_grad_matmul();
    test_grad_broadcast_add();
    test_grad_relu();
    test_grad_transpose();
    test_grad_softmax();
    test_grad_mlp();
    printf("all tests passed\n");
    return 0;
}
