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
# WARMUP matches bench.c's (not the original 10): this GPU idles down to
# ~210MHz between bursts and a short warmup doesn't consistently absorb the
# clock-ramp effect before the timed reps start (same finding that's in
# bench.c's own WARMUP comment) -- confirmed here too, torch_unfused's own
# number at S=128 ranged 0.014-0.077ms across repeated isolated runs with
# the old WARMUP=10 before this change.
WARMUP = 150
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


def run_size(S):
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

    # Peak allocator high-water mark for the unfused call, isolated to just
    # this call via reset_peak_memory_stats -- the caching allocator's own
    # view (see docs/research/tensorcore-and-fusion.md sec.4), not raw
    # kernel-necessary bytes, but the number an actual PyTorch user sees.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    unfused()
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    print(f"torch_unfused_peak_mem_bytes,{S},{S},{S},{peak_bytes}")
    sys.stdout.flush()


def main():
    # --size S: process-isolated single-size mode (docs/bench_report.html's
    # methodology section: one (kind,size) pair per process, fresh CUDA
    # context, no shared-process confound). Default (no args) still runs the
    # full sweep in one process for quick local iteration, same convention
    # as bench.c's --pair vs full-sweep modes -- its output isn't what gets
    # published, --size's is.
    print("kind,M,K,N,ms", file=sys.stdout)
    if len(sys.argv) == 3 and sys.argv[1] == "--size":
        run_size(int(sys.argv[2]))
        return
    for S in SIZES:
        run_size(S)


if __name__ == "__main__":
    main()
