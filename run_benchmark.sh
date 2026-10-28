#!/usr/bin/env bash
# Run the benchmark: 100 iterations, thread counts 1..max, best of 3 runs each.
# Usage: ./run_benchmark.sh [path/to/heat_matrix.csv]
set -e
cd "$(dirname "$0")"
CSV=${1:-heat_matrix.csv}
[ -x ./heat_diffusion ] || ./compile.sh

MAX=$(nproc)
THREADS="1"
for ((t=2; t<MAX; t*=2)); do THREADS="$THREADS,$t"; done
[ "$MAX" -gt 1 ] && THREADS="$THREADS,$MAX"

# Pin threads to physical cores for stable timings
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Threads tested: $THREADS"
./heat_diffusion --csv "$CSV" --iters 100 --threads "$THREADS" --repeat 3 \
                 --out results.csv --dump final_downsampled.csv | tee run_log.txt

command -v python3 >/dev/null && python3 plot_results.py || echo "(python3 not found: skip plots)"
