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
