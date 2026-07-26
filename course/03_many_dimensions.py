# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lesson 3 — More than one input
#
# Real models have many inputs. An aerofoil has dozens of shape parameters; a
# climate model has scores of tunable constants. Lessons 1 and 2 worked in one
# dimension, where everything is comfortable. This lesson is about what breaks
# when you add dimensions, and the two things you can do about it.
#
# The short version: the number of basis terms explodes, so you (a) choose a
# smaller basis, and (b) exploit the fact that most of its coefficients are
# nearly zero.

# %%
import os
from math import comb

import numpy as np
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax.numpy as jnp
import equadratures.jax as eqj

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def save(name):
    plt.savefig(os.path.join(FIGDIR, name), dpi=120, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## 3.1 Building a multivariate basis
#
# The construction is the obvious one. With one orthonormal family per input,
# a multivariate basis function is a product across dimensions,
#
# $$\Phi_{\mathbf{k}}(x_1,\dots,x_d) \;=\; p_{k_1}(x_1)\,p_{k_2}(x_2)\cdots p_{k_d}(x_d)$$
#
# and it inherits orthonormality automatically, provided the inputs are
# independent. A basis is therefore just a **set of multi-indices**
# $\mathbf{k} = (k_1,\dots,k_d)$: which products to keep.

# %%
idx = eqj.total_order_indices(dimensions=2, order=2)
print("total-order basis, d=2, order=2:")
for row in idx:
    print(f"   k = {tuple(int(v) for v in row)}   ->  p_{row[0]}(x1) * p_{row[1]}(x2)")

# %% [markdown]
# ## 3.2 Two ways to choose the index set
#
# * **Tensor grid** — every combination with each $k_i \le p$. Cardinality
#   $(p+1)^d$. Exponential in `d`.
# * **Total order** — keep only $\sum_i k_i \le p$. Cardinality
#   $\binom{d+p}{d}$. Still grows fast, but far more slowly.
#
# Total order drops the high-order *interaction* terms, which are usually the
# least important. It is the sensible default.

# %%
print(f"{'d':>3} {'order':>6} {'tensor':>12} {'total-order':>13}")
for d in (1, 2, 3, 5, 10):
    for p in (3, 5):
        print(f"{d:3d} {p:6d} {(p+1)**d:12,d} {comb(d+p, d):13,d}")

# %% [markdown]
# Look at `d=10, order=5`: a tensor grid wants 60 million terms; total order
# wants 3003. And 3003 is *still* more model runs than most people have.

# %%
ds = np.arange(1, 13)
plt.figure(figsize=(6.2, 3.6))
plt.semilogy(ds, [(5 + 1) ** d for d in ds], "o-", label="tensor (order 5)")
plt.semilogy(ds, [comb(d + 5, d) for d in ds], "s-", label="total order (5)")
plt.axhline(1000, color="k", ls="--", lw=1, label="a realistic run budget")
plt.xlabel("input dimension"); plt.ylabel("basis terms"); plt.legend()
plt.grid(alpha=.3, which="both"); plt.title("the curse of dimensionality, drawn")
save("03_curse.png")

# %% [markdown]
# ## 3.3 Fitting when you can afford the quadrature
#
# If the tensor grid is affordable, use **spectral projection**: each coefficient
# is a single integral, computed by the quadrature of lesson 2. It is exact when
# the rule is exact for the products involved.
#
# Our test function is a polynomial, so the surrogate should reproduce it to
# machine precision.

# %%
def model(Z):
    """1 + 2*x1 + 3*x1*x2 -- deliberately exactly representable."""
    return 1.0 + 2.0 * Z[:, 0] + 3.0 * Z[:, 0] * Z[:, 1]


rec = [eqj.uniform_recurrence(4), eqj.uniform_recurrence(4)]
idx = eqj.total_order_indices(2, 2)
X, W = eqj.tensor_quadrature(rec)

poly = eqj.Poly(rec, idx)
poly.fit_projection(X, model(X), W)

print(f"{X.shape[0]} model runs, {idx.shape[0]} basis terms")
print("coefficients:", np.round(np.array(poly.coefficients), 10))

Xt = jnp.asarray(np.random.default_rng(0).uniform(-1, 1, size=(200, 2)))
err = float(jnp.max(jnp.abs(poly.predict(Xt) - model(Xt))))
print("max error at 200 unseen points:", err)

# %% [markdown]
# Notice how few coefficients are non-zero. That is the observation the rest of
# this lesson exploits.

# %% [markdown]
# ## 3.4 Fitting from scattered runs: least squares
#
# More often you cannot dictate where the model was run — you have a pile of
# existing results. Then fit by least squares: build the design matrix `A` whose
# columns are the basis functions evaluated at your points, and solve
# `A c ≈ y`.

# %%
rng = np.random.default_rng(1)
Xs = jnp.asarray(rng.uniform(-1, 1, size=(60, 2)))
poly_ls = eqj.Poly(rec, idx)
poly_ls.fit(Xs, model(Xs))
print("least squares from 60 scattered runs, max error:",
      float(jnp.max(jnp.abs(poly_ls.predict(Xt) - model(Xt)))))

A = poly_ls.get_design(Xs)
print("design matrix shape (runs x basis terms):", A.shape)

# %% [markdown]
# Least squares needs **at least as many runs as basis terms**, and comfortably
# more for stability. In high dimensions that is exactly the budget you do not
# have.

# %% [markdown]
# ## 3.5 The way out: sparsity
#
# Here is the empirical fact that rescues the situation. For most physical
# models, **most coefficients are negligible**. The response is driven by a few
# main effects and a couple of interactions; the rest is noise.
#
# If the answer is sparse, you can recover it from *fewer runs than unknowns* —
# this is compressed sensing. The tool is an $\ell_1$ penalty, which drives
# coefficients to exactly zero rather than merely small:
#
# $$\min_c \; \tfrac12\|Ac - y\|_2^2 \;+\; \lambda\|c\|_1$$

# %%
# 28 basis terms, but only 20 model runs: least squares cannot do this.
rec6 = [eqj.uniform_recurrence(8), eqj.uniform_recurrence(8)]
idx6 = eqj.total_order_indices(2, 6)
rng = np.random.default_rng(7)
X20 = jnp.asarray(rng.uniform(-1, 1, size=(20, 2)))
Xtest = jnp.asarray(rng.uniform(-1, 1, size=(200, 2)))

print(f"{idx6.shape[0]} basis terms, {X20.shape[0]} model runs "
      f"-> under-determined by {idx6.shape[0] - X20.shape[0]}")

sparse = eqj.Poly(rec6, idx6)
c = sparse.fit_lasso(X20, model(X20), regularisation=1e-3, max_iter=8000)
print("non-zero coefficients:", int(jnp.count_nonzero(c)), "of", idx6.shape[0])
print("max test error       :", float(jnp.max(jnp.abs(sparse.predict(Xtest) - model(Xtest)))))

# %% [markdown]
# ## 3.6 The penalty is a dial, and it has a cost
#
# Larger `lambda` means sparser, but it also **shrinks** the surviving
# coefficients towards zero — a bias proportional to `lambda`. Watch both effects
# at once.

# %%
lams = np.geomspace(1e-5, 1e-1, 12)
nnz, errs = [], []
for lam in lams:
    p = eqj.Poly(rec6, idx6)
    cc = p.fit_lasso(X20, model(X20), float(lam), max_iter=8000)
    nnz.append(int(jnp.count_nonzero(cc)))
    errs.append(float(jnp.max(jnp.abs(p.predict(Xtest) - model(Xtest)))))

fig, ax1 = plt.subplots(figsize=(6.2, 3.6))
ax1.semilogx(lams, nnz, "o-", color="C0"); ax1.set_ylabel("non-zero terms", color="C0")
ax1.set_xlabel(r"penalty $\lambda$"); ax1.grid(alpha=.3)
ax2 = ax1.twinx(); ax2.loglog(lams, errs, "s-", color="C3")
ax2.set_ylabel("max test error", color="C3")
plt.title("sparsity and accuracy pull in opposite directions")
save("03_lasso_path.png")

# %% [markdown]
# ## 3.7 Getting the sparsity without the bias
#
# The standard fix is two-stage: let $\ell_1$ *choose* which terms matter, then
# re-fit those terms by ordinary least squares with no penalty at all. You keep
# the sparsity and throw away the shrinkage.

# %%
lam = 1e-2
biased = eqj.Poly(rec6, idx6)
biased.fit_lasso(X20, model(X20), lam, max_iter=8000)
debiased = eqj.Poly(rec6, idx6)
debiased.fit_lasso_debiased(X20, model(X20), lam, max_iter=8000)

print(f"at lambda = {lam}")
print("  l1 only  : error", float(jnp.max(jnp.abs(biased.predict(Xtest) - model(Xtest)))))
print("  debiased : error", float(jnp.max(jnp.abs(debiased.predict(Xtest) - model(Xtest)))))
print("  both keep", int(jnp.count_nonzero(debiased.coefficients)), "terms")

# %% [markdown]
# ## 3.8 An honest limitation
#
# Sparse recovery is not magic. It provably works when the columns of `A` are
# sufficiently *incoherent* — roughly, when no basis function looks like a
# combination of the others on your sample points. When that fails, $\ell_1$
# returns a superset of the true terms: it will not usually *miss* an important
# term, but it may keep some spurious ones.
#
# Practical reading: treat the recovered support as a shortlist to check, not a
# verdict. And prefer sample points that are well spread — the same lesson as
# lesson 1, in a new costume.

# %% [markdown]
# ## What to take away
#
# * A multivariate basis is a set of multi-indices; total order beats tensor
#   grids badly as dimension grows.
# * Spectral projection is exact but needs a quadrature grid you can afford.
# * Least squares needs more runs than basis terms.
# * $\ell_1$ / compressed sensing breaks that limit when the truth is sparse,
#   which for physical models it usually is.
# * The penalty trades sparsity against bias; debiasing recovers the accuracy.
#
# ## Exercises
#
# 1. For `d=6`, what order can you afford on a budget of 200 runs using a
#    total-order basis and least squares (say, twice as many runs as terms)?
# 2. Add noise to the model outputs and re-run §3.5. Does sparse recovery
#    survive?
# 3. Compare `fit_lasso` with `fit_ridge` at matched sparsity. Why does ridge
#    never produce exact zeros?
#
# Solutions below.

# %%
if __name__ == "__main__":
    print("\n--- Solutions ---")

    # 1. Largest p with 2*C(6+p,6) <= 200.
    d, budget = 6, 200
    best = max(p for p in range(1, 12) if 2 * comb(d + p, d) <= budget)
    print(f"1. d={d}: order {best} needs {comb(d+best, d)} terms "
          f"({2*comb(d+best, d)} runs at 2x); order {best+1} would need "
          f"{comb(d+best+1, d)} terms. So order {best}.")

    # 2. Noise degrades but does not destroy recovery; the penalty must rise
    #    to roughly the noise level.
    rng2 = np.random.default_rng(11)
    y_noisy = model(X20) + jnp.asarray(rng2.normal(0, 1e-2, size=X20.shape[0]))
    print("2. with 1e-2 noise on the outputs:")
    for lam in (1e-4, 1e-2, 5e-2):
        p = eqj.Poly(rec6, idx6)
        cc = p.fit_lasso(X20, y_noisy, lam, max_iter=8000)
        e = float(jnp.max(jnp.abs(p.predict(Xtest) - model(Xtest))))
        print(f"   lambda={lam:<6}: {int(jnp.count_nonzero(cc)):2d} terms, "
              f"test error {e:.4f}")
    print("   Recovery survives, but lambda must be raised towards the noise level.")

    # 3. Ridge penalises the SQUARE of each coefficient, whose gradient vanishes
    #    at zero, so nothing is ever pushed exactly to zero. The l1 penalty has a
    #    kink at zero and a non-vanishing gradient, which is what produces exact
    #    zeros.
    pr = eqj.Poly(rec6, idx6)
    pr.fit_ridge(X20, model(X20), 1e-2)
    pl = eqj.Poly(rec6, idx6)
    pl.fit_lasso(X20, model(X20), 1e-2, max_iter=8000)
    print(f"3. ridge non-zeros: {int(jnp.count_nonzero(pr.coefficients))} of "
          f"{idx6.shape[0]}; lasso non-zeros: "
          f"{int(jnp.count_nonzero(pl.coefficients))}")
    print("   Ridge shrinks smoothly (gradient of c^2 vanishes at 0); the l1 kink")
    print("   at the origin is what makes exact zeros possible.")
