# SymNMF Clustering

Final project for the Software Project course (0368-2161) at Tel Aviv University.

Implements clustering based on Symmetric Non-negative Matrix Factorization (SymNMF), and compares it against K-means using silhouette score.

## How it works

Given a set of points:
1. Build a similarity matrix `A` between all points.
2. Compute the diagonal degree matrix `D`.
3. Normalize: `W = D^(-1/2) A D^(-1/2)`.
4. Factorize `W` into `H` (non-negative, lower-dimensional) such that `W ≈ H·Hᵀ`, using an iterative update rule until convergence.
5. Each point's cluster = the column in `H` where it scores highest.

The heavy computation (matrices, iterative solver) is written in C for speed, and wrapped as a Python extension using the CPython C API, so it can be called from Python like a normal module.

## Files

- `symnmf.c` / `symnmf.h` — core algorithm, also runnable as a standalone CLI
- `symnmfmodule.c` — Python C extension wrapper
- `symnmf.py` — Python CLI
- `kmeans.py` — standalone K-means implementation
- `analysis.py` — runs both SymNMF and K-means, compares via silhouette score
- `setup.py` — builds the Python extension
- `Makefile` — builds the C executable
- `run_tests.sh` — test suite (checks output + memory leaks via Valgrind)

## Notes

- `H` is initialized randomly in `[0, 2*sqrt(mean(W)/k)]`.
- Update rule uses β = 0.5, stops at max iterations or once the change in `H` is small enough.
- C code checked for memory leaks with Valgrind.
