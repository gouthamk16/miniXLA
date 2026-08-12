#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "graph.h"
#include "optimizer.h"
#include "ptx.h"

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

int main(void) {
    test_const_and_eval_op();
    test_fused_exec();
    test_redundant_removal();
    test_constant_folding();
    test_fusion();
    test_dce_after_fusion();
    test_ptx_codegen();
    printf("all tests passed\n");
    return 0;
}
