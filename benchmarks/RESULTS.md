# equadratures.jax benchmarks

- platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`, Python 3.12.3
- jax 0.11.0 on cpu
- numpy 2.5.1
- mode: full

> **No GPU on this machine** -- `jaxlib` is the CPU build, so the
> accelerator half of deliverable D2 is *not* evidenced below.
> What is measured here (compilation, batching, adjoint scaling)
> is hardware-independent. GPU and `pmap` parity remain open.
### jit versus eager

Same JAX code, with and without compilation. `compile` is the one-off first-call cost; it is amortised over every later call.

| benchmark | baseline | equadratures.jax | speed-up | notes |
|---|---|---|---|---|
| gauss_quadrature (n=16) | 327.5 us | 102.4 us | 3.2x | 61.21 ms |
| gauss_quadrature (n=64) | 30.23 ms | 4.70 ms | 6.4x | 107.22 ms |
| gauss_quadrature (n=256) | 33.93 ms | 23.59 ms | 1.4x | 638.13 ms |
| design_matrix (4000 pts, 70 terms) | 18.94 ms | 409.2 us | 46.3x | 76.09 ms |

### vmap versus a Python loop

Batching an independent sweep: many measures, or many penalties, evaluated in one compiled call instead of one at a time.

| benchmark | baseline | equadratures.jax | speed-up | notes |
|---|---|---|---|---|
| gauss_quadrature x 512 measures | 80.21 ms | 3.76 ms | 21.3x | 78.03 ms |
| lasso path over 24 penalties | 71.23 ms | 31.82 ms | 2.2x | 140.33 ms |

### Reverse-mode gradient versus finite differences

Gradient of the surrogate variance with respect to every training observation. Finite differences need two extra solves per parameter; the adjoint needs one pass regardless. The speed-up is therefore expected to grow linearly with the parameter count, and does.

| benchmark | baseline | equadratures.jax | speed-up | notes |
|---|---|---|---|---|
| d(variance)/d(y), 16 parameters | 3.34 ms | 133.5 us | 25x | agree to 4.8e-10 |
| d(variance)/d(y), 64 parameters | 22.86 ms | 105.8 us | 216x | agree to 3.6e-11 |
| d(variance)/d(y), 256 parameters | 115.62 ms | 192.8 us | 600x | agree to 3.9e-12 |

### Classic NumPy versus JAX

Identical task, independent implementations (verified equal to ~1e-11 by tests/test_jax_parity_with_classic.py). The point is that the port costs nothing on CPU, not that it wins.

| benchmark | baseline | equadratures.jax | speed-up | notes |
|---|---|---|---|---|
| design_matrix (4000 pts, 15 terms) | 317.5 us | 245.3 us | 1.3x | - |
| gauss_quadrature (n=64) | 44.68 ms | 622.8 us | 71.7x | - |

