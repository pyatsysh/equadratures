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
| gauss_quadrature (n=16) | 332.2 us | 103.2 us | 3.2x | 77.89 ms |
| gauss_quadrature (n=64) | 838.1 us | 335.2 us | 2.5x | 111.47 ms |
| gauss_quadrature (n=256) | 6.00 ms | 4.65 ms | 1.3x | 113.08 ms |
| design_matrix (4000 pts, 70 terms) | 24.54 ms | 440.1 us | 55.8x | 96.34 ms |

### vmap versus a Python loop

Batching an independent sweep: many measures, or many penalties, evaluated in one compiled call instead of one at a time.

| benchmark | baseline | equadratures.jax | speed-up | notes |
|---|---|---|---|---|
| gauss_quadrature x 512 measures | 94.64 ms | 18.00 ms | 5.3x | 101.59 ms |
| lasso path over 24 penalties | 82.80 ms | 38.72 ms | 2.1x | 232.46 ms |

### Reverse-mode gradient versus finite differences

Gradient of the surrogate variance with respect to every training observation. Finite differences need two extra solves per parameter; the adjoint needs one pass regardless. The speed-up is therefore expected to grow linearly with the parameter count, and does.

| benchmark | baseline | equadratures.jax | speed-up | notes |
|---|---|---|---|---|
| d(variance)/d(y), 16 parameters | 4.06 ms | 124.8 us | 33x | agree to 4.8e-10 |
| d(variance)/d(y), 64 parameters | 29.66 ms | 105.4 us | 281x | agree to 3.6e-11 |
| d(variance)/d(y), 256 parameters | 120.21 ms | 142.0 us | 847x | agree to 3.9e-12 |

### Neural-operator layer

The integral operator in the polynomial basis. `explicit kernel` means forming `kappa(x_i, y_j)` and doing the quadrature sum directly, which is what a naive kernel method does and what `kernel_gram` returns; the spectral route never forms it. Note the first row: on a small grid the spectral route is no faster and can be **slower**, because its advantage is asymptotic in the discretisation rather than universal. It is kept in the table for that reason.

| benchmark | baseline | equadratures.jax | speed-up | notes |
|---|---|---|---|---|
| apply: 14 nodes, 200 query points | 146.2 us (explicit kernel) | 169.4 us | 0.9x | same answer to 5e-13 |
| apply: 50 nodes, 1000 query points | 252.2 us (explicit kernel) | 94.1 us | 2.7x | same answer to 4e-13 |
| apply: 120 nodes, 5000 query points | 1.87 ms (explicit kernel) | 209.5 us | 8.9x | same answer to 2e-13 |
| apply: 300 nodes, 20000 query points | 38.24 ms (explicit kernel) | 541.4 us | 70.6x | same answer to 4e-13 |
| vmap over 128 input functions (order 6) | 78.89 ms (python loop) | 63.2 us | 1248x | one function at a time is the obvious way to write it |
| d(loss)/dR, 49 parameters (order 6) | 8.67 ms (finite diff) | 16.9 us | 513x | agree to 1.1e-14 |
| vmap over 128 input functions (order 10) | 77.73 ms (python loop) | 70.4 us | 1104x | one function at a time is the obvious way to write it |
| d(loss)/dR, 121 parameters (order 10) | 26.42 ms (finite diff) | 12.9 us | 2048x | agree to 3.1e-14 |
| vmap over 128 input functions (order 16) | 84.51 ms (python loop) | 159.5 us | 530x | one function at a time is the obvious way to write it |
| d(loss)/dR, 289 parameters (order 16) | 83.89 ms (finite diff) | 22.6 us | 3712x | agree to 6.1e-14 |

### Classic NumPy versus JAX

Identical task, independent implementations (verified equal to ~1e-11 by tests/test_jax_parity_with_classic.py). The point is that the port costs nothing on CPU, not that it wins.

| benchmark | baseline | equadratures.jax | speed-up | notes |
|---|---|---|---|---|
| design_matrix (4000 pts, 15 terms) | 313.4 us | 206.0 us | 1.5x | - |
| gauss_quadrature (n=64) | 387.8 us | 458.0 us | 0.8x | - |

