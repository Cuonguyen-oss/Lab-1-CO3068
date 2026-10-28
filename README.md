# Lab 1 - Heat diffusion with OpenMP (CO3068)

## Files
| File | Purpose |
|---|---|
| `heat_diffusion.cpp` | Sequential + 3 OpenMP parallel convolutions, timing, verification |
| `compile.sh` | Compile script (`./compile.sh` for g++, `./compile.sh clang` for clang++) |
| `run_benchmark.sh` | Runs 100 iterations, threads 1..nproc, best of 3, then draws the plots |
| `plot_results.py` | Makes `benchmark.png` (time / speedup / efficiency) and `heatmap.png` |

> NOTE: speedup needs a machine with several CPU cores. In VirtualBox: Settings > System > Processor.

## Run
```bash
sudo apt install build-essential libomp-dev python3-matplotlib python3-numpy
cp /path/to/heat_matrix.csv .        # put the data next to the code
./compile.sh
./run_benchmark.sh                   # produces results.csv, run_log.txt, benchmark.png, heatmap.png
# or manually:
./heat_diffusion --iters 100 --threads 1,2,4,8 --repeat 3
```

## Algorithms
* **Sequential**: 3 nested loops, ping-pong buffers (`cur` -> `nxt`), padded array with a 1-cell halo = 30.
* **DUMMY parallel**: `parallel for collapse(2) schedule(static)`, fork/join every iteration, generic 3x3 loop (parallel but not optimised).
* **PRO parallel**: one persistent `parallel` region, `omp for schedule(static)` by rows, SIMD stencil using the kernel symmetry (3 multiplications instead of 9), `restrict` pointers; the implicit barrier separates iterations.

All versions use the same `conv_cell()` => results are verified equal to the sequential run (max error ~1e-8, only rounding order differs) (`MaxAbsErr`).
Sanity check: kernel sums to 1.0, so the mean temperature is conserved (41564.2397 before and after).
