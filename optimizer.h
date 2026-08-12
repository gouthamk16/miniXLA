#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "graph.h"

// Optimize the graph rooted at `root` in place. Runs the simplifying passes to
// a fixpoint, then frees orphaned nodes. Returns the (possibly new) root.
Node* optimize(Node* root);

#endif
