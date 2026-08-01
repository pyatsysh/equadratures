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

## 6. Neural operators on a polynomial basis

A neural operator learns a map between *functions*. Every architecture is a stack
of `v(x) = σ(W u(x) + (K u)(x) + b)` and they differ only in how the integral
operator `K` is represented: the Fourier neural operator makes `κ` diagonal in a
Fourier basis and applies it with an FFT. Here `κ` sits in the orthonormal
polynomial basis and the integral is a Gauss quadrature:

`κ(x, y) = Σ_jk R_jk φ_j(x) φ_k(y)`,  `(K u)(x) = Σ_jk R_jk φ_j(x) c_k`,
`c_k = Σ_i w_i φ_k(x_i) u(x_i)`

```python
layer = eqj.SpectralOperatorLayer(rec, indices, channels_in=1, channels_out=1)
params = layer.init_params(jax.random.PRNGKey(0))
v = layer.apply(u, params)                  # u sampled at layer.X
```

Three properties follow from the representation rather than from training, and
they are what makes the polynomial basis worth using here:

- **Interpretable.** `R` is the operator's matrix in a basis of known polynomial
  modes — which input mode feeds which output mode, and how strongly.
- **Exact on the span.** Any linear operator mapping the span into itself has an
  exact `R = ⟨φ_j, A φ_k⟩`. `derivative_operator_tensor` builds it for `d/dx`,
  and the layer then reproduces differentiation to ~1e-15 relative rather than
  approximating it — analytically, with no optimiser involved. *Learning* the
  same operator from data is a separate and weaker claim: gradient descent
  recovers it to ~1e-5 (anneal the learning rate; Adam is unstable once the
  gradient reaches round-off).
- **Discretisation-invariant, exactly.** The parameters live on modes, not on a
  grid, so evaluating on a finer quadrature returns the *same function* to
  round-off — not merely a similar one.

`mode="diagonal"` is the direct FNO analogue and, single-channel, is exactly the
Mercer kernel of section 5 with `R_k = θ_k²`. `mode="dense"` couples modes too,
which it must to represent a non-symmetric operator such as differentiation.

```python
R = eqj.derivative_operator_tensor(layer)   # exact d/dx, no training
stack = eqj.NeuralOperator([layer1, layer2])
```

One caveat, stated rather than buried: between layers the output is re-projected
onto the span, and `σ(v)` generally is not in it. A **linear** stack is exact; a
nonlinear one carries a truncation error per layer, controlled by the mode count.
The FNO truncates the same way.

When comparing a *learned* operator against a known one, use
`effective_spectral_tensor(layer, params)` and not the raw `params["spectral"]`.
The pointwise term acts as a multiple of the identity in mode space, so the two
blocks are not separately identifiable: measured in the test suite, the raw
tensor is off by ~1e-1 where the operator it defines is right.

Worked script: `examples/jax_neural_operator.py`.

## 7. Latent-variable models — learning the subspace that matters

Many models with dozens of inputs vary along only a handful of directions:
`f(x) ≈ g(Uᵀx)` with `U` a `d × r` orthonormal matrix whose columns are learned
linear combinations of the inputs. Two routes, answering different questions.

**With gradients** — `C = E[∇f ∇fᵀ]`, whose small eigenvalues mark directions
the model ignores. Classically the gradients are the obstacle; here they are one
`jax.grad` of the surrogate.

```python
eigenvalues, U = eqj.active_subspace(poly.predict, X, dimension=1)
```

**Without them** — `PolynomialRidge` learns `U` and the polynomial together from
function values alone, by **variable projection**: for any `U` the best
coefficients are a linear least squares, so they are eliminated in closed form
and the loss depends on `U` alone. Differentiating it means differentiating
*through* a least-squares solve.

```python
model = eqj.PolynomialRidge(dimensions=8, subspace_dimension=1, order=3)
model.fit(X, y)
model.predict(X_new)
```

The classic namespace does this with a hand-derived Gauss–Newton Jacobian, an
Armijo line search and two hand-tuned constants — its own source comment asks
*"How do we know these are the best values of gamma and beta?"*. Autodiff
removes both the Jacobian and the question. `U` is constrained by construction
(a sign-fixed QR), so no manifold optimiser is needed either.

On a genuine ridge function both routes recover the true subspace to ~1e-15, and
the active subspace agrees with the classic implementation's eigenvalues to
eight decimal places.

**Compare subspaces, never `U` itself.** `U` and `UQ` for orthogonal `Q` are the
same subspace and the same model, so an elementwise comparison measures the
rotation an optimiser happened to land on. Use `subspace_distance`. (Same trap as
the spectral/pointwise split in section 6, in different clothes.)

Worked script: `examples/jax_dimension_reduction.py`.

## 8. Bayesian inference — NUTS straight through the surrogate

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

The same holds for the neural-operator layer, where the spectral tensor `R` has
`n_modes²` entries and so finite differences scale badly: **513x** at 49
parameters, **2048x** at 121, **3712x** at 289, agreeing to ~1e-14.

Applying the operator has a second, structural saving — never forming the kernel
`κ(x_i, y_j)`. That one is **asymptotic, not universal**, and the benchmark keeps
the row that shows it losing:

| apply | explicit kernel | spectral | speed-up |
|---|---|---|---|
| 14 nodes, 200 query points | 146 us | 169 us | **0.9x** |
| 50 nodes, 1 000 query points | 252 us | 94 us | 2.7x |
| 120 nodes, 5 000 query points | 1.87 ms | 210 us | 8.9x |
| 300 nodes, 20 000 query points | 38.2 ms | 541 us | 70.6x |

Explicit costs `O(M·N)`; spectral costs `O((M+N)·n + n²)`. On a coarse grid the
overhead dominates and you should not bother.

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
`truncated_gaussian_density` · `PolynomialKernel`, `gp_nlml`, `gp_predict`,
`gp_predict_with_variance` · `lasso`, `lasso_debiased`, `lasso_path`,
`elastic_net`, `ridge`, `soft_threshold` · `SpectralOperatorLayer`,
`NeuralOperator`, `effective_spectral_tensor`, `linear_operator_tensor`,
`derivative_operator_tensor` · `PolynomialRidge`, `active_subspace`,
`gradient_covariance`, `orthonormalise`, `subspace_distance`.

Optional, imported separately (needs `[jax-bayes]`):
`equadratures.jax.inverse` — `surrogate_input_model`, `sparse_coefficient_model`,
`run_nuts`, `run_svi`, `summarise`, `to_arviz`, `posterior_samples`.
