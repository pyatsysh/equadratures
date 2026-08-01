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
# # Lesson 1 — Why polynomials?
#
# You have a model. Running it once is expensive: a CFD solve, a climate run, a
# lab experiment. You want to know things about it — its average output, how
# sensitive it is to each input, what happens at settings you have not tried —
# and you cannot afford to run it thousands of times.
#
# The standard move is to build a cheap stand-in, a **surrogate**, from a modest
# number of runs. This course builds surrogates out of polynomials. By the end of
# this lesson you will know why polynomials, why the *obvious* way of fitting
# them fails, and what to do instead.

# %%
import os

import numpy as np
import matplotlib
if not os.environ.get("DISPLAY"):        # headless: write files instead of windows
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax.numpy as jnp
import equadratures.jax as eqj

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def save(name):
    """Save the current figure; harmless in a notebook, useful as a script."""
    plt.savefig(os.path.join(FIGDIR, name), dpi=120, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## 1.1 The model we will pretend is expensive
#
# Throughout the lesson, this stands in for your simulation. It is smooth but not
# a polynomial, which is the usual situation.

# %%
def expensive_model(x):
    """A smooth, non-polynomial 'simulation' on [-1, 1]."""
    return np.exp(-x) * np.sin(3.0 * x) + 0.5 * x


xs = np.linspace(-1, 1, 400)
plt.figure(figsize=(6, 3.2))
plt.plot(xs, expensive_model(xs), lw=2)
plt.title("the model we are allowed to sample, but not often")
plt.xlabel("x"); plt.ylabel("f(x)"); plt.grid(alpha=.3)
save("01_model.png")

# %% [markdown]
# ## 1.2 Why polynomials at all?
#
# Because of a theorem you can rely on: **Weierstrass** says any continuous
# function on a closed interval can be approximated as closely as you like by a
# polynomial. For *smooth* functions the approximation improves extremely fast —
# faster than any power of the number of terms.
#
# That is the promise. The catch is in *how* you find the coefficients.

# %% [markdown]
# ## 1.3 The obvious approach, and how it fails
#
# The obvious approach: pick `n+1` equally spaced points, and find the degree-`n`
# polynomial passing exactly through them. This is interpolation, and for
# equally spaced points it is a trap.
#
# Watch what happens as we raise the degree on Runge's function,
# `1 / (1 + 25 x^2)`.

# %%
def runge(x):
    return 1.0 / (1.0 + 25.0 * x ** 2)


plt.figure(figsize=(6.5, 3.6))
plt.plot(xs, runge(xs), "k", lw=2, label="truth")
for n in (5, 9, 15):
    nodes = np.linspace(-1, 1, n + 1)
    coef = np.polyfit(nodes, runge(nodes), n)
    plt.plot(xs, np.polyval(coef, xs), lw=1.4, label=f"degree {n}")
plt.ylim(-1.5, 2.0); plt.legend(); plt.grid(alpha=.3)
plt.title("equally spaced interpolation gets WORSE with degree (Runge)")
save("01_runge.png")

# %% [markdown]
# The error near the ends grows without bound as the degree increases. More data
# made the answer worse. This is not a rounding problem; it is real.
#
# Two things are wrong, and they have two different fixes:
#
# 1. **The points are in the wrong places.** Equally spaced points are a bad
#    choice for high-degree interpolation. Clustering points towards the ends
#    fixes it — and the good clusterings turn out to be exactly the *quadrature
#    nodes* of lesson 2.
# 2. **The basis is badly conditioned.** Writing a polynomial as
#    `c0 + c1 x + c2 x^2 + ...` gives a fitting matrix that becomes numerically
#    singular fast, because `x^7` and `x^9` look nearly identical on `[-1, 1]`.
#    The fix is an **orthogonal** basis.

# %% [markdown]
# ## 1.4 Fix one: better points
#
# Let us keep the same degrees but put the points where the Gauss rule wants
# them. `equadratures.jax` gives us those directly.

# %%
plt.figure(figsize=(6.5, 3.6))
plt.plot(xs, runge(xs), "k", lw=2, label="truth")
for n in (5, 9, 15):
    alpha, beta, mu0 = eqj.legendre_recurrence(n + 1)
    nodes, _ = eqj.gauss_quadrature(alpha, beta, mu0)
    nodes = np.array(nodes)
    coef = np.polyfit(nodes, runge(nodes), n)
    plt.plot(xs, np.polyval(coef, xs), lw=1.4, label=f"degree {n}")
plt.ylim(-1.5, 2.0); plt.legend(); plt.grid(alpha=.3)
plt.title("the same degrees, at Gauss points: convergence restored")
save("01_gauss_points.png")

# %% [markdown]
# Same function, same degrees, same amount of data. Only the *locations*
# changed. This is the single most useful practical lesson in the course: **where
# you sample matters as much as how much you sample.**

# %% [markdown]
# ## 1.5 Fix two: an orthogonal basis
#
# Now the conditioning problem. Instead of the *monomials* `1, x, x^2, ...` we
# use polynomials `p_0, p_1, p_2, ...` that are **orthonormal** with respect to a
# weight `w(x)`:
#
# $$\int p_i(x)\, p_j(x)\, w(x)\, dx = \delta_{ij}$$
#
# Read `w(x)` as "the probability density of the input". If your input is uniform
# on `[-1, 1]`, the orthonormal family is the (normalised) **Legendre**
# polynomials; if it is a standard normal, they are the **Hermite** polynomials.
#
# Let us look at them, and then check the orthogonality claim numerically.

# %%
alpha, beta, mu0 = eqj.uniform_recurrence(8)
P = np.array(eqj.orthonormal_polynomials(jnp.asarray(xs), alpha, beta, mu0, 4))

plt.figure(figsize=(6.5, 3.6))
for k in range(5):
    plt.plot(xs, P[k], lw=1.5, label=f"$p_{k}$")
plt.legend(ncol=5, fontsize=8); plt.grid(alpha=.3)
plt.title("orthonormal polynomials for a uniform input on [-1, 1]")
save("01_orthonormal.png")

# %% [markdown]
# The orthogonality is not decoration — it is what makes the fit well behaved.
# Check it. We integrate `p_i p_j` using a quadrature rule (lesson 2 explains
# where the rule comes from; for now, trust it).

# %%
nodes, weights = eqj.gauss_quadrature(alpha, beta, mu0)
Pn = np.array(eqj.orthonormal_polynomials(nodes, alpha, beta, mu0, 4))
gram = (Pn * weights) @ Pn.T
print("Gram matrix (should be the identity):")
print(np.round(gram, 12))
print("max deviation from identity:", np.abs(gram - np.eye(5)).max())

# %% [markdown]
# Exactly the identity, to machine precision. Compare that with the monomial
# basis on the same points, whose Gram matrix is nearly singular:

# %%
V = np.vander(np.array(nodes), 5, increasing=True)
gram_mono = (V.T * weights) @ V
print("monomial Gram condition number :", np.linalg.cond(gram_mono))
print("orthonormal Gram condition number:", np.linalg.cond(gram))

# %% [markdown]
# A condition number of ~1 means the fitting problem is as well posed as it can
# be. That is the whole argument for orthogonal bases.

# %% [markdown]
# ## 1.6 Putting it together: a first surrogate
#
# `equadratures.jax` wraps all of this. You describe the input, pick a maximum
# degree, evaluate the model at the points it asks for, and fit.

# %%
DEGREE = 12
rec = [eqj.uniform_recurrence(DEGREE + 1)]
idx = eqj.total_order_indices(dimensions=1, order=DEGREE)

X, W = eqj.tensor_quadrature(rec)         # the points to run your model at
y = expensive_model(np.array(X)[:, 0])    # ... and the runs themselves

poly = eqj.Poly(rec, idx)
poly.fit_projection(X, jnp.asarray(y), W)

pred = np.array(poly.predict(jnp.asarray(xs).reshape(-1, 1)))
truth = expensive_model(xs)
print(f"{len(y)} model runs, degree {DEGREE}")
print("max error over [-1, 1]:", np.abs(pred - truth).max())

plt.figure(figsize=(6.5, 3.6))
plt.plot(xs, truth, "k", lw=2, label="truth")
plt.plot(xs, pred, "--", lw=2, label="surrogate")
plt.plot(np.array(X)[:, 0], y, "o", ms=5, label="model runs")
plt.legend(); plt.grid(alpha=.3); plt.title("a surrogate from 13 model runs")
save("01_surrogate.png")

# %% [markdown]
# ## 1.7 How accuracy grows with effort
#
# For a smooth function, adding degrees pays off *exponentially* — the hallmark
# of spectral methods, and the reason this approach is worth the trouble.

# %%
degrees, errors = list(range(2, 25, 2)), []
for d in degrees:
    r = [eqj.uniform_recurrence(d + 1)]
    Xd, Wd = eqj.tensor_quadrature(r)
    p = eqj.Poly(r, eqj.total_order_indices(1, d))
    p.fit_projection(Xd, jnp.asarray(expensive_model(np.array(Xd)[:, 0])), Wd)
    e = np.abs(np.array(p.predict(jnp.asarray(xs).reshape(-1, 1))) - truth).max()
    errors.append(max(e, 1e-17))

plt.figure(figsize=(6, 3.4))
plt.semilogy(degrees, errors, "o-")
plt.xlabel("polynomial degree"); plt.ylabel("max error")
plt.title("straight line on a log axis = exponential convergence")
plt.grid(alpha=.3, which="both")
save("01_convergence.png")

print("error at degree 2 :", errors[0])
print("error at degree 24:", errors[-1])

# %% [markdown]
# Note where it flattens out: once the error reaches ~1e-15 there is nothing left
# to win, because double-precision arithmetic has run out of digits. Recognising
# that floor — rather than chasing it with more degrees — is part of using these
# methods well.

# %% [markdown]
# ## What to take away
#
# * Polynomials approximate smooth functions extremely efficiently.
# * **Where you sample matters.** Equally spaced points fail at high degree;
#   Gauss points do not.
# * **Which basis you use matters.** Monomials are numerically hopeless;
#   orthonormal polynomials are perfectly conditioned.
# * The weight `w(x)` defining orthogonality is the input's probability density.
#   That link is what turns approximation into uncertainty quantification in
#   lesson 4.
#
# ## Exercises
#
# 1. Change `expensive_model` to `lambda x: np.abs(x)` and re-run the convergence
#    study. Does the error still fall exponentially? Why not?
# 2. Redo Section 1.4 with Chebyshev points, `cos(pi k / n)`. Do they also fix Runge?
# 3. At which degree does the surrogate in Section 1.6 first reach an error below 1e-6?
#
# Solutions below.

# %%
if __name__ == "__main__":
    print("\n--- Solutions ---")

    # 1. |x| is continuous but not differentiable at 0. Smoothness is what buys
    #    exponential convergence, so we expect only algebraic convergence here.
    errs_abs = []
    for d in degrees:
        r = [eqj.uniform_recurrence(d + 1)]
        Xd, Wd = eqj.tensor_quadrature(r)
        p = eqj.Poly(r, eqj.total_order_indices(1, d))
        p.fit_projection(Xd, jnp.abs(Xd[:, 0]), Wd)
        errs_abs.append(np.abs(np.array(p.predict(jnp.asarray(xs).reshape(-1, 1)))
                               - np.abs(xs)).max())
    ratio = errs_abs[0] / errs_abs[-1]
    print(f"1. |x|: error fell only {ratio:.1f}x from degree {degrees[0]} to "
          f"{degrees[-1]} (smooth case fell {errors[0]/errors[-1]:.1e}x).")
    print("   The kink at x=0 caps convergence at an algebraic rate.")

    # 2. Chebyshev points also cluster at the ends, so they also fix Runge.
    n = 15
    cheb = np.cos(np.pi * np.arange(n + 1) / n)
    err_cheb = np.abs(np.polyval(np.polyfit(cheb, runge(cheb), n), xs)
                      - runge(xs)).max()
    equi = np.linspace(-1, 1, n + 1)
    err_equi = np.abs(np.polyval(np.polyfit(equi, runge(equi), n), xs)
                      - runge(xs)).max()
    print(f"2. degree {n} on Runge: equispaced max error {err_equi:.3f}, "
          f"Chebyshev {err_cheb:.4f}. Yes, clustering is what matters.")

    # 3. First degree below 1e-6.
    hit = next((d for d, e in zip(degrees, errors) if e < 1e-6), None)
    print(f"3. First degree with error < 1e-6: {hit}")
