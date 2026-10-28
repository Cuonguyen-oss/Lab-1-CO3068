#!/usr/bin/env bash
# Compile Lab 1 (heat diffusion with OpenMP).
# Usage: ./compile.sh            -> g++   (default)
#        ./compile.sh clang      -> clang++
set -e
cd "$(dirname "$0")"

CXX=${1:-g++}
[ "$CXX" = "clang" ] && CXX=clang++

if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "Compiler '$CXX' not found. Install:  sudo apt install build-essential libomp-dev"
    exit 1
fi

# -O3                : same optimisation level (-O3) for sequential and parallel code
# -ffp-contract=off  : forbid FMA fusion (keeps rounding identical between versions)
$CXX -std=c++17 -O3 -fopenmp -ffp-contract=off -Wall -Wextra \
     heat_diffusion.cpp -o heat_diffusion

echo "Build OK -> ./heat_diffusion"
