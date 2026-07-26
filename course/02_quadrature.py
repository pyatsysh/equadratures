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
# # Lesson 2 — Integration you can trust
#
# Almost everything you want from a model is an integral. The average output is
# an integral. The variance is an integral. So are the sensitivity indices of
# lesson 4, and the coefficients of the surrogate itself.
#
# So the question "how do I integrate a function I can only *sample*?" is not a
# side issue. This lesson answers it, and shows that the answer was hiding
# inside lesson 1 all along: the good sample locations and the good integration
# rule are the same points.

# %%
import os

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
# ## 2.1 The shape of every quadrature rule
#
# Every method in this lesson has the same form: sample at some **nodes**
# `x_1..x_n`, and take a weighted sum,
#
# $$\int f(x)\, w(x)\, dx \;\approx\; \sum_{i=1}^{n} w_i\, f(x_i)$$
#
# Methods differ only in *where the nodes go* and *what the weights are*. That is
# the entire design space, and it is smaller than it looks.

# %% [markdown]
# ## 2.2 The baseline everyone reaches for: Monte Carlo
#
# Sample at random, average. It is simple, it works in any number of dimensions,
# and its error falls like $1/\sqrt{N}$ — which is *slow*. To gain one decimal
# place you need a hundred times more model runs.

# %%
rng = np.random.default_rng(0)
f = lambda x: np.exp(-x) * np.sin(3.0 * x) + 0.5 * x

# reference value on [-1,1] against the uniform *probability* density (w = 1/2)
ref_nodes, ref_w = eqj.gauss_quadrature(*eqj.uniform_recurrence(80))
reference = float(jnp.sum(ref_w * f(np.array(ref_nodes))))
print("reference value of E[f(X)], X ~ U(-1,1):", reference)

Ns = np.array([10, 100, 1000, 10_000, 100_000])
mc_err = []
for N in Ns:
    trials = [abs(np.mean(f(rng.uniform(-1, 1, N))) - reference) for _ in range(20)]
    mc_err.append(np.mean(trials))

plt.figure(figsize=(6, 3.4))
plt.loglog(Ns, mc_err, "o-", label="Monte Carlo")
plt.loglog(Ns, mc_err[0] * np.sqrt(Ns[0] / Ns), "k--", label=r"$1/\sqrt{N}$")
plt.xlabel("model runs"); plt.ylabel("mean error"); plt.legend(); plt.grid(alpha=.3, which="both")
plt.title("Monte Carlo: honest, general, and slow")
save("02_monte_carlo.png")

# %% [markdown]
# ## 2.3 Gauss quadrature: the same accuracy for far fewer runs
#
# If you get to *choose* where to sample, you can do enormously better. An
# `n`-point **Gauss rule** integrates every polynomial up to degree `2n-1`
# exactly. That is the best possible: `n` nodes and `n` weights are `2n` free
# numbers, and matching `2n` moments uses them all.
#
# Let us verify the exactness claim rather than assert it. We integrate $x^p$
# against the uniform density on `[-1,1]`, whose exact value is `0` for odd `p`
# and `1/(p+1)` for even `p`.

# %%
n = 5
alpha, beta, mu0 = eqj.uniform_recurrence(n)
nodes, weights = eqj.gauss_quadrature(alpha, beta, mu0)

print(f"{n}-point Gauss rule: exact up to degree {2*n-1}\n")
print(" p |     quadrature |          exact |    error")
for p in range(0, 2 * n + 2):
    approx = float(jnp.sum(weights * nodes ** p))
    exact = 0.0 if p % 2 else 1.0 / (p + 1)
    flag = "   <-- degree 2n-1, still exact" if p == 2 * n - 1 else ""
    flag = "   <-- 2n, exactness ends here" if p == 2 * n else flag
    print(f"{p:2d} | {approx: 14.10f} | {exact: 14.10f} | {abs(approx-exact):.2e}{flag}")

# %% [markdown]
# Exact to machine precision right up to degree `2n-1`, and wrong at `2n`. The
# rule is not accidentally better than advertised — that sharpness is what lets
# you *reason* about the accuracy you are buying.

# %% [markdown]
# ## 2.4 Where do the nodes come from? An eigenvalue problem
#
# This is the elegant part. The orthonormal polynomials of lesson 1 obey a
# three-term recurrence,
#
# $$x\,p_k(x) = \beta_k\, p_{k-1}(x) + \alpha_k\, p_k(x) + \beta_{k+1}\, p_{k+1}(x)$$
#
# Collect the coefficients into a symmetric tridiagonal **Jacobi matrix**. Then:
#
# * the Gauss **nodes** are its eigenvalues;
# * the Gauss **weights** are $\mu_0$ times the squared first components of its
#   eigenvectors.
#
# That is the Golub–Welsch algorithm, and it is all `gauss_quadrature` does.
# Because an eigendecomposition is differentiable, so is the whole rule — which
# is the hinge the rest of this course swings on.

# %%
J = np.array(eqj.jacobi_matrix(alpha, beta))
print("Jacobi matrix (symmetric, tridiagonal):")
print(np.round(J, 4))

eigvals, eigvecs = np.linalg.eigh(J)
print("\neigenvalues          :", np.round(eigvals, 8))
print("gauss_quadrature says:", np.round(np.array(nodes), 8))
print("\nweights from eigenvectors:", np.round(mu0 * eigvecs[0] ** 2, 8))
print("gauss_quadrature says    :", np.round(np.array(weights), 8))

# %% [markdown]
# ## 2.5 The weights are a probability distribution
#
# For a probability measure the weights are non-negative and sum to one. They
# *are* a discrete distribution — a set of representative scenarios with
# probabilities attached. That reading is worth holding on to: a quadrature rule
# is a principled way to replace a continuous input distribution with a handful
# of weighted cases.

# %%
print("weights sum to:", float(jnp.sum(weights)), " (probability measure)")
print("all positive  :", bool(jnp.all(weights > 0)))

plt.figure(figsize=(6, 3.2))
plt.stem(np.array(nodes), np.array(weights), basefmt=" ")
plt.xlabel("node"); plt.ylabel("weight")
plt.title("a 5-point Gauss rule, read as a discrete distribution")
plt.grid(alpha=.3)
save("02_weights.png")

# %% [markdown]
# ## 2.6 Comparing the two on equal terms
#
# How many model runs does each method need for a given accuracy? This is the
# comparison that decides whether the extra machinery is worth it.

# %%
gauss_ns = np.arange(2, 15)
gauss_err = []
for gn in gauss_ns:
    nd, wt = eqj.gauss_quadrature(*eqj.uniform_recurrence(int(gn)))
    gauss_err.append(max(abs(float(jnp.sum(wt * f(np.array(nd)))) - reference), 1e-17))

plt.figure(figsize=(6.2, 3.6))
plt.loglog(Ns, mc_err, "o-", label="Monte Carlo")
plt.loglog(gauss_ns, gauss_err, "s-", label="Gauss")
plt.xlabel("model runs"); plt.ylabel("error"); plt.legend(); plt.grid(alpha=.3, which="both")
plt.title("smooth 1-D integrand: Gauss wins by orders of magnitude")
save("02_gauss_vs_mc.png")

print("Gauss error with 10 runs      :", gauss_err[8])
print("Monte Carlo error with 100000 :", mc_err[-1])

# %% [markdown]
# **Be careful how far you generalise this.** Gauss dominates for *smooth*
# integrands in *low* dimensions. Monte Carlo's $1/\sqrt{N}$ does not care about
# dimension, while a tensor-product Gauss rule needs $n^d$ points. Lesson 3 is
# about what to do when `d` is large enough for that to matter.

# %% [markdown]
# ## 2.7 When you need a node in a particular place
#
# Sometimes a node has to sit somewhere specific: a boundary condition, a
# reference operating point, a design constraint at the edge of the domain.
#
# * **Radau** rules fix one node, and are exact to degree `2n-2`.
# * **Lobatto** rules fix both endpoints, and are exact to degree `2n-3`.
#
# You pay one degree per constraint. Nothing is free, but the price is small and
# known.

# %%
a5, b5, m5 = eqj.legendre_recurrence(5)
for label, (nd, wt), deg in [
    ("Gauss  ", eqj.gauss_quadrature(a5, b5, m5), 2 * 5 - 1),
    ("Radau  ", eqj.radau_quadrature(a5, b5, m5, -1.0), 2 * 5 - 2),
    ("Lobatto", eqj.lobatto_quadrature(a5, b5, m5, -1.0, 1.0), 2 * 5 - 3),
]:
    print(f"{label} (exact to degree {deg}): nodes {np.round(np.array(nd), 4)}")

# %% [markdown]
# ## 2.8 Distributions other than uniform
#
# Your inputs are rarely uniform. The recurrence — and therefore the rule —
# depends on the input density. Two are analytic and built in; anything else you
# supply as a density and the family is constructed numerically.

# %%
# Gaussian input: nodes spread over the real line, weights are normal probabilities
hn, hw = eqj.gauss_quadrature(*eqj.hermite_recurrence(6))
print("Gauss-Hermite nodes :", np.round(np.array(hn), 4))
print("E[X^2] for X~N(0,1):", float(jnp.sum(hw * hn ** 2)), " (exact 1.0)")

# An arbitrary density: Beta(2,3) on [0,1].
p = eqj.DensityParameter(eqj.beta_density(2.0, 3.0), 0.0, 1.0)
ba, bb, bmu = p.recurrence(6)
bn, bw = eqj.gauss_quadrature(ba, bb, bmu)
print("\nBeta(2,3) nodes     :", np.round(np.array(bn), 4))
print("E[X]  computed:", float(jnp.sum(bw * bn)), " exact:", 2 / 5)
print("Var[X] computed:", float(jnp.sum(bw * (bn - 2 / 5) ** 2)),
      " exact:", 2 * 3 / ((2 + 3) ** 2 * (2 + 3 + 1)))

# %% [markdown]
# There is a neat fact buried in that construction: for a probability measure the
# very first recurrence coefficients *are* the mean and the variance —
# `alpha[0] = E[X]` and `beta[0]**2 = Var[X]`.

# %%
print("alpha[0]   :", float(ba[0]), " = E[X]")
print("beta[0]**2 :", float(bb[0] ** 2), " = Var[X]")

# %% [markdown]
# ## What to take away
#
# * Everything you want is an integral; quadrature is how you get it.
# * An `n`-point Gauss rule is exact to degree `2n-1` — provably the best
#   possible, and sharp.
# * The nodes and weights come from an eigendecomposition of the Jacobi matrix.
#   That is why they are differentiable.
# * The weights form a probability distribution over representative scenarios.
# * Gauss beats Monte Carlo decisively for smooth low-dimensional problems, and
#   that advantage erodes as dimension grows.
# * Prescribed-node rules cost one degree of exactness per fixed node.
#
# ## Exercises
#
# 1. How many Gauss points integrate $x^{11}$ exactly? Check it.
# 2. Integrate the non-smooth `|x|` with Gauss rules of growing size. Does the
#    error fall as fast as it did for the smooth integrand?
# 3. Build a rule for a truncated Gaussian on `[-2, 2]` and compute `E[X^2]`.
#    Should it be larger or smaller than 1?
#
# Solutions below.

# %%
if __name__ == "__main__":
    print("\n--- Solutions ---")

    # 1. Need 2n-1 >= 11, so n >= 6.
    for nn in (5, 6):
        nd, wt = eqj.gauss_quadrature(*eqj.uniform_recurrence(nn))
        err = abs(float(jnp.sum(wt * nd ** 11)) - 0.0)
        print(f"1. n={nn}: error integrating x^11 = {err:.2e} "
              f"({'exact' if err < 1e-14 else 'NOT exact'})")
    print("   2n-1 >= 11 needs n >= 6, and that is exactly where it becomes exact.")

    # 2. |x| has a kink; exactness arguments need smoothness.
    print("2. |x| on [-1,1], exact value 0.5:")
    for nn in (4, 8, 16, 32):
        nd, wt = eqj.gauss_quadrature(*eqj.uniform_recurrence(nn))
        err = abs(float(jnp.sum(wt * jnp.abs(nd))) - 0.5)
        print(f"   n={nn:3d}: error {err:.3e}")
    print("   Only algebraic decay: the kink, not the rule, is the bottleneck.")

    # 3. Truncation removes the tails, which carry the largest |x|, so E[X^2] < 1.
    pt = eqj.DensityParameter(eqj.truncated_gaussian_density(0.0, 1.0), -2.0, 2.0)
    ta, tb, tmu = pt.recurrence(8)
    tn, tw = eqj.gauss_quadrature(ta, tb, tmu)
    m2 = float(jnp.sum(tw * tn ** 2))
    print(f"3. E[X^2] on a N(0,1) truncated to [-2,2] = {m2:.6f}")
    print(f"   Smaller than 1 ({m2 < 1}): truncation discards the heavy-|x| tails.")
