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

Arbitrary distributions? Hand it a **density** and the orthonormal family is
built numerically — and differentiably, including with respect to the
distribution's own parameters:

```python
p = eqj.DensityParameter(eqj.beta_density(a=2.0, b=3.0), lower=0.0, upper=1.0)
alpha, beta, mu0 = p.recurrence(6)      # alpha[0] = E[X], beta[0]**2 = Var[X]
```

So you can ask *how sensitive is my output to how I modelled the input?* — a
derivative the classic namespace cannot produce at all:

```python
jax.grad(lambda a: variance_of_surrogate_under(eqj.beta_density(a, 3.0)))(2.0)
```

The interval is discretised on **Gauss–Legendre** nodes, not an equispaced grid.
Besides being far more accurate, the nodes are strictly *interior*, so densities
that blow up at an endpoint still work — `Beta(0.5, 0.5)` returns `inf` from
classic's equispaced pdf grid but is handled here. `stieltjes_recurrence(nodes,
weights, n)` remains available if you want to supply a discretised measure
directly.

Need a node *pinned* somewhere — a boundary condition, a design constraint?
Radau fixes one, Lobatto fixes both ends, at a cost of one and two degrees of
exactness respectively. Both are differentiable in the prescribed nodes too.

```python
nodes, weights = eqj.radau_quadrature(alpha, beta, mu0, end=-1.0)        # 2n-2
nodes, weights = eqj.lobatto_quadrature(alpha, beta, mu0, -1.0, 1.0)     # 2n-3
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

`fit` (least squares), `fit_ridge` (Tikhonov), `fit_lasso`,
`fit_lasso_debiased` and `fit_elastic_net` are also available — see §4.

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

## 4. Sparse fits — compressed sensing, with gradients

Fewer samples than basis terms? Use an `l1` penalty. The classic
`compressed-sensing` solver hands the problem to `cvxpy`, which is opaque to
autodiff; here it is solved natively (FISTA) and differentiated **implicitly**,
by differentiating the optimality conditions rather than unrolling the
iterations.

```python
poly.fit_lasso(X, y, regularisation=1e-3)          # sparse, exact zeros
poly.fit_lasso_debiased(X, y, 1e-3)                # ... then re-fit on the support
poly.fit_elastic_net(X, y, l1=1e-3, l2=1e-4)       # + ridge on correlated columns
```

That buys three things over unrolling: gradients are **exact** rather than
iteration-count-dependent, they cost **one linear solve** on the active set
regardless of `max_iter`, and memory stays flat. Because the penalties are
themselves differentiable inputs, they can be *learned* on a validation loss
instead of grid-searched:

```python
jax.grad(lambda l1, l2: validation_mse(l1, l2), argnums=(0, 1))(1e-3, 1e-3)
```

`eqj.lasso_path` sweeps a whole penalty path in one `vmap`ped call.

## 5. Learnable polynomial (Mercer) kernel — the "random polynomial kernel"

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

## 6. Bayesian inference — NUTS straight through the surrogate

Gradient-based samplers need `d(log density)/d(parameters)` through the forward
model. That derivative is exactly what the classic namespace cannot give, so
this is the clearest illustration of what differentiability is *for*.

```bash
pip install equadratures[jax-bayes]     # + numpyro, arviz
```

```python
from equadratures.jax import inverse as eqi

# inverse UQ: recover unknown inputs from noisy measurements
model = eqi.surrogate_input_model(forward, y_obs, lower=[-1, -1], upper=[1, 1])
mcmc = eqi.run_nuts(model, num_warmup=500, num_samples=1500)
eqi.summarise(mcmc)["x"]           # posterior mean, sd, 90% interval
eqi.to_arviz(mcmc)                 # r_hat, ESS, trace plots

# Bayesian compressed sensing: the posterior sibling of fit_lasso
model = eqi.sparse_coefficient_model(A, y, tau_scale=0.5)
```

`run_svi` fits a mean-field guide instead, when MCMC is more than you need.
Worked end-to-end script: `examples/jax_bayesian_inverse.py`.

## Performance

`benchmarks/jax_benchmarks.py` measures compilation, batching and adjoint
scaling; `benchmarks/RESULTS.md` holds a reference run. The headline is the
gradient cost — reverse mode returns the whole gradient for roughly the price of
one extra evaluation, while finite differences pay per parameter:

| `d(variance)/d(y)` | finite differences | `jax.grad` | speed-up |
|---|---|---|---|
| 16 parameters | 3.34 ms | 133 us | 25x |
| 64 parameters | 22.9 ms | 106 us | 216x |
| 256 parameters | 116 ms | 193 us | 600x |

*(CPU-only box; the accelerator half of the story is untested here — see the
caveat printed by the benchmark script.)*

## Design & validation philosophy

Every primitive is validated by a **sum-rule / exactness** check *and* an
**autodiff-vs-finite-difference** check (see `tests/test_jax_*.py`): quadrature
exactness to degree `2n-1`, orthonormality through the quadrature
(`⟨p_i,p_j⟩ = δ_ij`), Stieltjes recovering the analytic Legendre recurrence,
exact polynomial recovery with correct moments/Sobol', and gradient correctness
throughout.

On top of that, `tests/test_jax_parity_with_classic.py` checks this namespace
against the **classic NumPy implementation** at every level — recurrence
coefficients (including Stieltjes on an arbitrary Beta measure), Gauss points and
weights, the design matrix, fitted coefficients, mean/variance, Sobol' indices
and predictions — all agreeing to ~1e-11. Two conventions differ deliberately and
are documented there: classic stores the *monic* recurrence (ours are the square
roots), and classic's tensor-grid basis enumerates multi-indices in a different
order (for total-order bases the two coincide).

## API

`gauss_quadrature`, `radau_quadrature`, `lobatto_quadrature`, `jacobi_matrix` ·
`legendre_recurrence`, `uniform_recurrence`, `hermite_recurrence`,
`stieltjes_recurrence` · `orthonormal_polynomials` · `total_order_indices`,
`tensor_grid_indices`, `design_matrix` · `Poly`, `tensor_quadrature` ·
`Parameter`, `DensityParameter`, `density_recurrence`, `beta_density`,
`truncated_gaussian_density` · `PolynomialKernel`, `gp_nlml`, `gp_predict` · `lasso`,
`lasso_debiased`, `lasso_path`, `elastic_net`, `ridge`, `soft_threshold`.

Optional, imported separately (needs `[jax-bayes]`):
`equadratures.jax.inverse` — `surrogate_input_model`, `sparse_coefficient_model`,
`run_nuts`, `run_svi`, `summarise`, `to_arviz`, `posterior_samples`.
