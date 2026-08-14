#!/bin/sh
# Build and run targets for MiniXLA.
#
# Not a Makefile: there is no object-file step here (every target compiles
# straight from sources), and the two toolchains are different compilers
# rather than one compiler with different flags -- gcc builds anything that
# never touches CUDA, nvcc builds anything that links the GPU runtime (it
# uses MSVC as the host compiler on Windows and takes this C99 as-is). That
# leaves make nothing to manage, and `make` isn't installed on the machine
# this is developed on anyway.
set -e

CORE="graph.c tensor.c optimizer.c"
CFLAGS="-O2 -Wall"

build_demo()       { gcc $CFLAGS -o demo       main.c       $CORE -lm; }
build_tests()      { gcc $CFLAGS -o tests      tests.c      $CORE ptx.c autodiff.c -lm; }
build_gpu_test()   { nvcc        -o gpu_test   gpu_test.c   $CORE ptx.c runtime.c gpu_exec.c -lcuda; }
build_bench()      { nvcc -O2    -o bench      bench.c      $CORE ptx.c gpu_exec.c -lcuda -lcublas; }
build_bench_loop() { nvcc -O2    -o bench_loop bench_loop.c $CORE ptx.c gpu_exec.c -lcuda -lcublas; }

case "${1:-test}" in
  demo)       build_demo ;;
  tests)      build_tests ;;
  gpu-test)   build_gpu_test ;;
  bench)      build_bench ;;
  bench-loop) build_bench_loop ;;
  all)        build_demo; build_tests; build_gpu_test; build_bench ;;
  # Both suites. The CPU and GPU paths execute the same optimized graphs, so
  # a change that only passes one side is not verified.
  test)       build_tests; build_gpu_test; ./tests; ./gpu_test ;;
  clean)      rm -f demo tests gpu_test bench bench_loop *.exe *.exp *.lib *.obj *.pdb ;;
  *) echo "usage: $0 {demo|tests|gpu-test|bench|bench-loop|all|test|clean}" >&2; exit 1 ;;
esac
