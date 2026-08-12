"""PyTorch (CUDA, eager) comparison points for the MiniXLA benchmark report.

Same sizes, same warmup/min-of-N methodology as bench.c, so the numbers are
comparable. torch.compile is NOT used here -- it requires Triton, which has
no working Windows install on this machine (verified: TritonMissing at
compile time). Eager mode is what a real user gets on this platform without
extra tooling, and is the honest baseline: it's cuBLAS/cuDNN-backed under
the hood, same as MiniXLA's raw-matmul comparison point, but issues each op
in the graph (matmul, add, relu) as a separate kernel launch -- exactly the
unfused case MiniXLA's operator fusion exists to avoid.
"""
import torch
import time
import sys

assert torch.cuda.is_available(), "CUDA not available"
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

SIZES = [128, 256, 512, 1024, 2048]
WARMUP = 10
REPS = 20


def timed_min(fn, warmup=WARMUP, reps=REPS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        if ms < best:
            best = ms
    return best


def main():
    print("kind,M,K,N,ms", file=sys.stdout)
    for S in SIZES:
        torch.manual_seed(S)
        a = torch.randn(S, S, device=device, dtype=torch.float32)
        b = torch.randn(S, S, device=device, dtype=torch.float32)
        c = torch.randn(S, S, device=device, dtype=torch.float32)

        # Raw matmul -- comparable to bench.c's cublas/minixla_gpu-matmul-only.
        ms_matmul = timed_min(lambda: torch.matmul(a, b))
        print(f"torch_matmul,{S},{S},{S},{ms_matmul:.4f}")

        # Unfused relu(a@b + c): 3 kernel launches (matmul, add, relu) --
        # comparable to MiniXLA's single fused kernel for the same op chain.
        def unfused():
            return torch.relu(torch.matmul(a, b) + c)
        ms_unfused = timed_min(unfused)
        print(f"torch_unfused_matmul_add_relu,{S},{S},{S},{ms_unfused:.4f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
