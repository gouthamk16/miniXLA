#ifndef PTX_H
#define PTX_H

#include "graph.h"

#define PTX_TILE 16   // default shared-memory tile side length

// Emit PTX with an explicit tile size (must be 8, 16, or 32).
char* emit_ptx_tiled(const Node* fused, int tile);

// Emit PTX with the default tile (PTX_TILE).
char* emit_ptx(const Node* fused);

#endif
