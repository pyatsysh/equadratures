# `equadratures.jax` — auto-differentiable equadratures

A JAX-native, **differentiable** backend for equadratures: quadrature,
orthogonal-polynomial approximation, uncertainty quantification, and *learnable*
polynomial kernels — all `jax.grad`-able, `jax.jit`-able and `jax.vmap`-able.

This is a **parallel namespace**: the classic NumPy `equadratures` API is
untouched. Everything here is pure/functional JAX, so you can differentiate
through quadrature, fitting and UQ outputs, and train polynomial models and
kernels by gradient descent.

```bash
pip install equadratures[jax]        # core (jax)
pip install equadratures[jax-learn]  # + optax, for kernel/GP training
```

> float64 is enabled on import (quadrature exactness needs double precision).

## 1. Differentiable Gauss quadrature

Nodes and weights come from the Golub–Welsch eigendecomposition of the Jacobi
matrix — and because `jnp.linalg.eigh` is differentiable, so are they.

```python
import equadratures.jax as eqj
import jax, jax.numpy as jnp

alpha, beta, mu0 = eqj.legendre_recurrence(8)   # uniform on [-1, 1]
nodes, weights = eqj.gauss_quadrature(alpha, beta, mu0)
# 8-point rule integrates degree <= 15 exactly:
approx = jnp.sum(weights * nodes**4)            # ~ integral of x^4 = 2/5
```

Arbitrary distributions? Discretise the measure and use the Stieltjes recurrence
(differentiable w.r.t. the nodes/weights, hence w.r.t. distribution parameters):

```python
alpha, beta, mu0 = eqj.stieltjes_recurrence(nodes, weights, n=6)
```

## 2. Uncertainty quantification with `Poly`

Fit an orthonormal polynomial surrogate and read off differentiable moments and
Sobol' sensitivity indices.

```python
rec = [eqj.uniform_recurrence(4), eqj.uniform_recurrence(4)]   # 2 uniform inputs
idx = eqj.total_order_indices(dimensions=2, order=2)
X, W = eqj.tensor_quadrature(rec)

f = lambda Z: 1.0 + 2.0*Z[:, 0] + 3.0*Z[:, 0]*Z[:, 1]
poly = eqj.Poly(rec, idx)
poly.fit_projection(X, f(X), W)                 # spectral projection

poly.mean()             # 1.0
poly.variance()         # 7/3
poly.sobol_indices()    # [4/7, 0]   (first-order)
poly.total_sobol_indices()  # [1, 3/7]
```

`fit` (least squares) and `fit_ridge` (Tikhonov, with a differentiable
regularisation) are also available.

## 3. Differentiate through *anything*

The pipeline is end-to-end differentiable — w.r.t. inputs, data, and
distribution parameters:

```python
# sensitivity of the surrogate to its input
jax.grad(lambda x: poly.predict(x[None, :])[0])(jnp.array([0.3, -0.4]))

# d(output variance)/d(training data)
jax.grad(lambda y: (lambda p: (p.fit_projection(X, y, W), p.variance())[1])(eqj.Poly(rec, idx)))(f(X))

# d E[g(X)] / d(distribution bound)   (via Parameter's differentiable map)
```

## 4. Learnable polynomial (Mercer) kernel — the "random polynomial kernel"

`k(x, x') = Σ_k θ_k² φ_k(x) φ_k(x')` on the orthonormal features: PSD by
construction, interpretable (weights on known polynomial modes), and *learnable*
via its log-spectrum.

```python
import optax
kernel = eqj.PolynomialKernel(rec=[eqj.uniform_recurrence(11)],
                              indices=eqj.total_order_indices(1, 10))
params = {"log_theta": kernel.default_log_theta(), "log_noise": jnp.log(1e-2)}
loss = lambda p: eqj.gp_nlml(kernel, Xtr, ytr, p["log_theta"], p["log_noise"])

opt = optax.adam(5e-2); state = opt.init(params)
step = jax.jit(jax.value_and_grad(loss))
for _ in range(300):                     # train the kernel spectrum + noise
    _, g = step(params)
    updates, state = opt.update(g, state, params)
    params = optax.apply_updates(params, updates)

pred = eqj.gp_predict(kernel, Xtr, ytr, Xte, params["log_theta"], params["log_noise"])
```

## Design & validation philosophy

Every primitive is validated by a **sum-rule / exactness** check *and* an
**autodiff-vs-finite-difference** check (see `tests/test_jax_*.py`): quadrature
exactness to degree `2n-1`, orthonormality through the quadrature
(`⟨p_i,p_j⟩ = δ_ij`), Stieltjes recovering the analytic Legendre recurrence,
exact polynomial recovery with correct moments/Sobol', and gradient correctness
throughout.

## API

`gauss_quadrature`, `jacobi_matrix` · `legendre_recurrence`, `uniform_recurrence`,
`hermite_recurrence`, `stieltjes_recurrence` · `orthonormal_polynomials` ·
`total_order_indices`, `tensor_grid_indices`, `design_matrix` · `Poly`,
`tensor_quadrature` · `Parameter` · `PolynomialKernel`, `gp_nlml`, `gp_predict`.
