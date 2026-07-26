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
# # Lesson 4 — Reading the surrogate
#
# So far the surrogate has been a fast stand-in for a slow model. This lesson is
# about the other thing it is: **an interpretable summary of how your model
# behaves**.
#
# Once you have the coefficients, the mean, the variance and a full sensitivity
# analysis are available without any further model runs — and without sampling.
# They are already in the numbers you have.

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
# ## 4.1 The one idea this lesson rests on
#
# Because the basis is **orthonormal with respect to the input distribution**,
# the coefficients are not just fitting parameters. They are the *variance
# decomposition of your model*.
#
# Write the surrogate as $f(x) \approx \sum_k c_k \Phi_k(x)$. Then, with
# $\Phi_0 = 1$ and every other $\Phi_k$ having zero mean:
#
# $$\mathbb{E}[f] = c_0, \qquad \mathrm{Var}[f] = \sum_{k>0} c_k^2$$
#
# The mean is one coefficient. The variance is Parseval's identity. No sampling.

# %%
def model(Z):
    """f = 1 + 2*x1 + 3*x1*x2 on independent uniform inputs."""
    return 1.0 + 2.0 * Z[:, 0] + 3.0 * Z[:, 0] * Z[:, 1]


rec = [eqj.uniform_recurrence(4), eqj.uniform_recurrence(4)]
idx = eqj.total_order_indices(2, 2)
X, W = eqj.tensor_quadrature(rec)

poly = eqj.Poly(rec, idx)
poly.fit_projection(X, model(X), W)

print("multi-index    coefficient")
for k, c in zip(idx, np.array(poly.coefficients)):
    print(f"  {tuple(int(v) for v in k)}        {c: .6f}")

print("\nmean     =", float(poly.mean()), "  (analytic 1)")
print("variance =", float(poly.variance()), "  (analytic 7/3 =", 7 / 3, ")")

# %% [markdown]
# We can check those against brute force. Monte Carlo agrees — slowly, and to
# three digits, having spent 200 000 model runs to match something we got from
# 25.

# %%
rng = np.random.default_rng(0)
samples = rng.uniform(-1, 1, size=(200_000, 2))
vals = model(samples)
print("Monte Carlo mean    :", vals.mean(), " from 200,000 runs")
print("Monte Carlo variance:", vals.var())
print("surrogate used      :", X.shape[0], "runs")

# %% [markdown]
# ## 4.2 Where does the variance come from? Sobol' indices
#
# Variance alone tells you the output moves. Sensitivity analysis tells you
# *which inputs move it*. Group the basis terms by which variables they involve:
#
# * terms in $x_1$ alone contribute the **main effect** of $x_1$;
# * terms in $x_1$ and $x_2$ contribute their **interaction**;
# * and so on.
#
# Dividing each group's summed squared coefficients by the total variance gives
# the **Sobol' indices**. First-order $S_i$ is the fraction explained by $x_i$
# acting alone; total-effect $T_i$ is the fraction involving $x_i$ at all. The
# gap between them *is* the interaction.

# %%
S = np.array(poly.sobol_indices())
T = np.array(poly.total_sobol_indices())
print("first-order S =", np.round(S, 6), "  (analytic [4/7, 0] =", [4 / 7, 0], ")")
print("total-effect T =", np.round(T, 6), "  (analytic [1, 3/7] =", [1, 3 / 7], ")")

x = np.arange(2)
plt.figure(figsize=(5.4, 3.4))
plt.bar(x - 0.19, S, 0.38, label="first order $S_i$")
plt.bar(x + 0.19, T, 0.38, label="total effect $T_i$")
plt.xticks(x, ["$x_1$", "$x_2$"]); plt.ylabel("fraction of variance")
plt.legend(); plt.grid(alpha=.3, axis="y")
plt.title("the gap between the bars is interaction")
save("04_sobol.png")

# %% [markdown]
# Read that plot carefully, because it is the whole point of sensitivity
# analysis:
#
# * $S_2 = 0$. On its own, $x_2$ does nothing.
# * $T_2 = 3/7$. Yet $x_2$ is involved in 43% of the variance.
#
# **A one-at-a-time sensitivity study would have concluded $x_2$ is irrelevant
# and fixed it at its nominal value.** That would have been badly wrong: $x_2$
# matters entirely through its interaction with $x_1$. This is the failure mode
# that variance-based sensitivity analysis exists to prevent.

# %% [markdown]
# ## 4.3 A more realistic example
#
# The Ishigami function is the standard sensitivity-analysis test case, chosen
# because it has strong interactions and known analytic indices. Inputs are
# uniform on $[-\pi, \pi]$.

# %%
A_ISH, B_ISH = 7.0, 0.1


def ishigami(Z):
    """Z columns are already scaled to [-pi, pi]."""
    x1, x2, x3 = Z[:, 0], Z[:, 1], Z[:, 2]
    return jnp.sin(x1) + A_ISH * jnp.sin(x2) ** 2 + B_ISH * x3 ** 4 * jnp.sin(x1)


ORDER = 12
rec3 = [eqj.uniform_recurrence(ORDER + 1)] * 3
idx3 = eqj.total_order_indices(3, ORDER)
Xq, Wq = eqj.tensor_quadrature([eqj.uniform_recurrence(ORDER + 2)] * 3)
Xphys = Xq * np.pi                      # map [-1,1]^3 -> [-pi,pi]^3

poly3 = eqj.Poly(rec3, idx3)
poly3.fit_projection(Xq, ishigami(Xphys), Wq)

# analytic values (standard results for the Ishigami function)
var_analytic = A_ISH ** 2 / 8 + B_ISH * np.pi ** 4 / 5 + B_ISH ** 2 * np.pi ** 8 / 18 + 0.5
v1 = 0.5 * (1 + B_ISH * np.pi ** 4 / 5) ** 2
v2 = A_ISH ** 2 / 8
v13 = B_ISH ** 2 * np.pi ** 8 * (1.0 / 18 - 1.0 / 50)

print(f"basis terms: {idx3.shape[0]},  quadrature runs: {Xq.shape[0]}")
print("variance  computed:", float(poly3.variance()), " analytic:", var_analytic)
print("S computed:", np.round(np.array(poly3.sobol_indices()), 5))
print("S analytic:", np.round([v1 / var_analytic, v2 / var_analytic, 0.0], 5))
print("T computed:", np.round(np.array(poly3.total_sobol_indices()), 5))
print("T analytic:", np.round([(v1 + v13) / var_analytic, v2 / var_analytic,
                               v13 / var_analytic], 5))

# %% [markdown]
# Again the interesting entry is $x_3$: first-order index zero, total effect
# clearly non-zero. It acts only through its interaction with $x_1$ — which is
# exactly how the function was built.

# %%
S3 = np.array(poly3.sobol_indices())
T3 = np.array(poly3.total_sobol_indices())
x = np.arange(3)
plt.figure(figsize=(5.8, 3.4))
plt.bar(x - 0.19, S3, 0.38, label="first order")
plt.bar(x + 0.19, T3, 0.38, label="total effect")
plt.xticks(x, ["$x_1$", "$x_2$", "$x_3$"]); plt.ylabel("fraction of variance")
plt.legend(); plt.grid(alpha=.3, axis="y"); plt.title("Ishigami sensitivity")
save("04_ishigami.png")

# %% [markdown]
# ## 4.4 Check your surrogate before you trust its indices
#
# Every number above is a property of the *surrogate*. If the surrogate is a poor
# fit, its variance decomposition describes something that is not your model.
# Always hold back some runs and look at the error.

# %%
rng = np.random.default_rng(2)
Xv = jnp.asarray(rng.uniform(-1, 1, size=(500, 3)))
pred = poly3.predict(Xv)
true = ishigami(Xv * np.pi)
rel = float(jnp.linalg.norm(pred - true) / jnp.linalg.norm(true))
print("relative L2 error on 500 held-out points:", rel)

plt.figure(figsize=(4.2, 4.0))
plt.plot(np.array(true), np.array(pred), ".", ms=4, alpha=.6)
lims = [float(true.min()), float(true.max())]
plt.plot(lims, lims, "k--", lw=1)
plt.xlabel("model"); plt.ylabel("surrogate"); plt.grid(alpha=.3)
plt.title(f"held-out check (rel. L2 = {rel:.2e})")
save("04_parity.png")

# %% [markdown]
# ## 4.5 Convergence of the indices
#
# A useful habit: recompute the indices at increasing order and watch them
# settle. If they are still moving, your order is too low.

# %%
orders = [4, 6, 8, 10, 12]
hist = []
for o in orders:
    r = [eqj.uniform_recurrence(o + 1)] * 3
    Xo, Wo = eqj.tensor_quadrature([eqj.uniform_recurrence(o + 2)] * 3)
    p = eqj.Poly(r, eqj.total_order_indices(3, o))
    p.fit_projection(Xo, ishigami(Xo * np.pi), Wo)
    hist.append(np.array(p.total_sobol_indices()))
hist = np.array(hist)

plt.figure(figsize=(6, 3.4))
for i in range(3):
    plt.plot(orders, hist[:, i], "o-", label=f"$T_{i+1}$")
plt.xlabel("polynomial order"); plt.ylabel("total-effect index")
plt.legend(); plt.grid(alpha=.3); plt.title("indices should stop moving")
save("04_convergence.png")

# %% [markdown]
# ## What to take away
#
# * Orthonormality makes the coefficients a variance decomposition: the mean is
#   $c_0$, the variance is $\sum_{k>0} c_k^2$.
# * Sobol' indices come free from coefficients you already have — no extra runs,
#   no sampling.
# * $S_i \ll T_i$ means the input matters *through interactions*. One-at-a-time
#   studies miss this, and that is why they are dangerous.
# * The indices describe the surrogate. Validate the surrogate first.
#
# ## Exercises
#
# 1. Set `B_ISH = 0` and recompute. What happens to $T_3$, and why?
# 2. What fraction of the Ishigami variance is *pure interaction* (not captured
#    by any first-order index)?
# 3. Fit the Ishigami function at order 4 and compare its indices with the
#    order-12 answer. Would you have drawn the same conclusions?
#
# Solutions below.

# %%
if __name__ == "__main__":
    print("\n--- Solutions ---")

    # 1. With B = 0 the x3 term disappears entirely, so x3 becomes inert.
    def ishigami_b0(Z):
        x1, x2 = Z[:, 0], Z[:, 1]
        return jnp.sin(x1) + A_ISH * jnp.sin(x2) ** 2

    r = [eqj.uniform_recurrence(11)] * 3
    Xo, Wo = eqj.tensor_quadrature([eqj.uniform_recurrence(12)] * 3)
    p = eqj.Poly(r, eqj.total_order_indices(3, 10))
    p.fit_projection(Xo, ishigami_b0(Xo * np.pi), Wo)
    print("1. B=0 -> T =", np.round(np.array(p.total_sobol_indices()), 6))
    print("   T3 collapses to ~0: with B=0, x3 appears nowhere in the function,")
    print("   so it has neither a main effect nor an interaction.")

    # 2. Pure interaction = 1 - sum of first-order indices.
    interaction = 1.0 - float(np.sum(S3))
    print(f"2. sum of first-order indices = {float(np.sum(S3)):.4f}, so "
          f"{100*interaction:.1f}% of the variance is pure interaction.")

    # 3. Low order under-resolves the x3^4 term.
    r4 = [eqj.uniform_recurrence(5)] * 3
    X4, W4 = eqj.tensor_quadrature([eqj.uniform_recurrence(6)] * 3)
    p4 = eqj.Poly(r4, eqj.total_order_indices(3, 4))
    p4.fit_projection(X4, ishigami(X4 * np.pi), W4)
    print("3. order 4  T =", np.round(np.array(p4.total_sobol_indices()), 4))
    print("   order 12 T =", np.round(T3, 4))
    print("   The ranking survives, but the magnitudes are off; quote indices")
    print("   only once they have stopped moving with order.")
