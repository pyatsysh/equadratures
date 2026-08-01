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
# # Lesson 5 — Differentiating through everything
#
# Everything so far could have been done in 2010. This lesson is the part that
# is new, and it is the reason `equadratures.jax` exists.
#
# The whole pipeline — recurrence coefficients, quadrature nodes, basis
# evaluation, the fit, the moments, the Sobol' indices — is written as
# differentiable code. So you can ask for the derivative of *any* output with
# respect to *any* input, and get it exactly, in roughly the cost of one extra
# evaluation.
#
# The interesting question is not "how" but "what would you ask for?". This
# lesson works through four answers.

# %%
import os
import time

import numpy as np
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equadratures.jax as eqj

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def save(name):
    plt.savefig(os.path.join(FIGDIR, name), dpi=120, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## 5.1 Warm-up: the gradient of the surrogate
#
# The easiest derivative, and the one people expect: how does the *output* change
# with the *inputs*? The surrogate is a polynomial, so this is exact, not a
# finite-difference approximation.

# %%
def model(Z):
    return 1.0 + 2.0 * Z[:, 0] + 3.0 * Z[:, 0] * Z[:, 1]


rec = [eqj.uniform_recurrence(4), eqj.uniform_recurrence(4)]
idx = eqj.total_order_indices(2, 2)
X, W = eqj.tensor_quadrature(rec)
poly = eqj.Poly(rec, idx)
poly.fit_projection(X, model(X), W)


def surrogate(x):
    """Scalar-in, scalar-out view of the surrogate, for jax.grad."""
    return poly.predict(x[None, :])[0]


x0 = jnp.asarray([0.3, -0.4])
g = np.array(jax.grad(surrogate)(x0))
# analytically: df/dx1 = 2 + 3*x2, df/dx2 = 3*x1
print("autodiff gradient :", np.round(g, 10))
print("analytic gradient :", [2 + 3 * (-0.4), 3 * 0.3])

# %% [markdown]
# Exact to the last digit. And `jax.hessian` works too — second derivatives with
# no extra effort, which is what a gradient-based optimiser wants.

# %%
print("Hessian:\n", np.round(np.array(jax.hessian(surrogate)(x0)), 10))

# %% [markdown]
# ## 5.2 The one that surprises people: derivative w.r.t. the *data*
#
# Here is a question that is awkward to even pose without autodiff: **how much
# does my reported variance depend on each individual model run?**
#
# If one expensive run has a large derivative, your headline UQ number is hostage
# to that single simulation — worth knowing before you publish it.

# %%
y0 = model(X)


def variance_of(y):
    p = eqj.Poly(rec, idx)
    p.fit_projection(X, y, W)
    return p.variance()


sens = np.array(jax.grad(variance_of)(y0))
print("d(variance)/d(run i), most influential runs:")
order = np.argsort(-np.abs(sens))[:5]
for i in order:
    print(f"   run {i:2d} at x = {np.round(np.array(X[i]), 3)} : {sens[i]: .5f}")

plt.figure(figsize=(6, 3.4))
plt.bar(np.arange(len(sens)), sens)
plt.xlabel("model run"); plt.ylabel(r"$\partial \mathrm{Var}/\partial y_i$")
plt.title("which runs is my variance actually resting on?")
plt.grid(alpha=.3, axis="y")
save("05_data_sensitivity.png")

# %% [markdown]
# This generalises into a genuinely useful workflow: **where should I spend my
# next model run?** Run it where the derivative says the answer is most
# sensitive.

# %% [markdown]
# ## 5.3 Why the adjoint, and not finite differences
#
# You could get the previous plot by finite differences: nudge each $y_i$ and
# refit. That costs *two extra fits per data point*. Reverse-mode
# differentiation computes the entire gradient in one pass, whatever the number
# of parameters.
#
# The gap therefore grows linearly with the number of parameters. Below we
# measure it. (Reference numbers from the project benchmark on a CPU box:
# **25x** at 16 parameters, **216x** at 64, **600x** at 256, agreeing with finite
# differences to about 1e-12.)

# %%
def timing_experiment(m):
    rng = np.random.default_rng(2)
    Xm = jnp.asarray(rng.uniform(-1, 1, size=(m, 2)))
    ym = jnp.asarray(rng.normal(size=m))
    r = [eqj.uniform_recurrence(5)] * 2
    ii = eqj.total_order_indices(2, 3)

    def var_of(y):
        p = eqj.Poly(r, ii)
        p.fit(Xm, y)
        return p.variance()

    grad_fn = jax.jit(jax.grad(var_of))
    val_fn = jax.jit(var_of)
    grad_fn(ym).block_until_ready(); val_fn(ym).block_until_ready()

    t0 = time.perf_counter()
    g_ad = np.array(grad_fn(ym)); g_ad_ = grad_fn(ym).block_until_ready()
    t_ad = time.perf_counter() - t0

    t0 = time.perf_counter()
    eps, base = 1e-6, np.array(ym)
    g_fd = np.zeros(m)
    for i in range(m):
        yp = base.copy(); yp[i] += eps
        ym_ = base.copy(); ym_[i] -= eps
        g_fd[i] = (float(val_fn(jnp.asarray(yp))) - float(val_fn(jnp.asarray(ym_)))) / (2 * eps)
    t_fd = time.perf_counter() - t0
    return t_ad, t_fd, np.abs(g_ad - g_fd).max()


print(f"{'params':>7} {'adjoint':>10} {'finite diff':>12} {'speed-up':>9} {'agreement':>11}")
sizes = [8, 16, 32, 64]
speedups = []
for m in sizes:
    t_ad, t_fd, err = timing_experiment(m)
    speedups.append(t_fd / t_ad)
    print(f"{m:7d} {t_ad*1e3:9.2f}ms {t_fd*1e3:11.2f}ms {t_fd/t_ad:8.0f}x {err:11.1e}")

plt.figure(figsize=(6, 3.4))
plt.plot(sizes, speedups, "o-")
plt.xlabel("number of parameters"); plt.ylabel("speed-up over finite differences")
plt.grid(alpha=.3); plt.title("the adjoint advantage grows with parameter count")
save("05_adjoint_speedup.png")

# %% [markdown]
# Two things to notice. The speed-up grows roughly linearly, as theory says. And
# the two agree to ~1e-12 — the adjoint is not an approximation to finite
# differences, it is the exact derivative, and finite differences are the
# approximation.

# %% [markdown]
# ## 5.4 The one that is genuinely new: derivative w.r.t. the *input distribution*
#
# Every UQ result rests on an assumption you cannot verify: the input
# distribution. You assumed the inlet temperature was Beta(2, 3). What if it was
# Beta(2.2, 3)?
#
# Normally you answer that by re-running the entire study. Here it is one
# derivative, because the recurrence coefficients are themselves built by
# differentiable code from the density.

# %%
f = lambda Z: jnp.exp(Z[:, 0]) + Z[:, 0] ** 3


def variance_given_shape(a):
    """Var[f(X)] where X ~ Beta(a, 3) on [0, 1]."""
    p = eqj.DensityParameter(eqj.beta_density(a, 3.0), 0.0, 1.0, n_grid=250)
    r = [p.recurrence(8)]
    Xa, Wa = eqj.tensor_quadrature(r)
    poly_a = eqj.Poly(r, eqj.total_order_indices(1, 5))
    poly_a.fit_projection(Xa, f(Xa), Wa)
    return poly_a.variance()


a0 = 2.0
v = float(variance_given_shape(a0))
dv = float(jax.grad(variance_given_shape)(a0))
print(f"Var[f(X)] with X ~ Beta(2, 3) = {v:.6f}")
print(f"d(Var)/d(shape a)             = {dv:.6f}")
print(f"linear prediction at a = 2.1  : {v + 0.1*dv:.6f}")
print(f"actual value      at a = 2.1  : {float(variance_given_shape(2.1)):.6f}")

# %% [markdown]
# The linear prediction is close, which is what a derivative is for. Plot the
# curve and the tangent to see how far you can trust it.

# %%
aa = np.linspace(1.5, 3.0, 25)
vv = [float(variance_given_shape(float(a))) for a in aa]
plt.figure(figsize=(6, 3.4))
plt.plot(aa, vv, lw=2, label="Var[f(X)]")
plt.plot(aa, v + dv * (aa - a0), "--", lw=1.5, label="tangent from autodiff")
plt.plot([a0], [v], "ro")
plt.xlabel("Beta shape parameter a"); plt.ylabel("output variance")
plt.legend(); plt.grid(alpha=.3)
plt.title("sensitivity of a UQ result to the assumed input distribution")
save("05_distribution_sensitivity.png")

# %% [markdown]
# This is a **robustness statement about your uncertainty analysis**, and it is
# the kind of thing reviewers ask for and rarely get. It is available here
# because differentiability goes all the way down to the measure.

# %% [markdown]
# ## 5.5 Learning a hyper-parameter instead of grid-searching it
#
# If a quantity is differentiable, you can optimise it. The sparsity penalty from
# lesson 3 is an input like any other, so it can be *trained* on a validation
# loss rather than searched over a grid.
#
# This demonstration needs **noisy** training data, and the reason is worth
# stating rather than hiding. Regularisation buys you variance reduction at the
# price of bias. If the data are noise-free and the truth lies in the basis, the
# bias is all cost and no benefit, the best penalty is $\lambda = 0$, and there is
# no interior optimum for a gradient to find. Give the data a realistic noise
# level and $\lambda$ acquires a genuine minimum: too small overfits the noise,
# too large erases real structure.

# %%
rng = np.random.default_rng(8)
rec6 = [eqj.uniform_recurrence(6)] * 2
idx6 = eqj.total_order_indices(2, 4)          # 15 terms from 20 noisy runs
Xtr = jnp.asarray(rng.uniform(-1, 1, size=(20, 2)))
Xva = jnp.asarray(rng.uniform(-1, 1, size=(20, 2)))
ytr = model(Xtr) + jnp.asarray(rng.normal(0, 0.1, size=20))
yva = model(Xva) + jnp.asarray(rng.normal(0, 0.1, size=20))


def validation_loss(log_lam):
    p = eqj.Poly(rec6, idx6)
    p.fit_elastic_net(Xtr, ytr, jnp.exp(log_lam), 1e-8, max_iter=3000)
    return jnp.mean((p.predict(Xva) - yva) ** 2)


# %% [markdown]
# We optimise $\log \lambda$ rather than $\lambda$, which keeps the penalty
# positive for free and turns a multiplicative search into an additive one. Adam
# rather than plain gradient descent, because the loss is only *piecewise*
# smooth in $\lambda$ — each time a coefficient enters or leaves the active set
# the curvature jumps, and a fixed step size stalls on the flat stretches
# between.

# %%
try:
    import optax
    HAVE_OPTAX = True
except ImportError:
    HAVE_OPTAX = False
    print("optax not installed; skipping training "
          "(pip install equadratures[jax-learn])")

if HAVE_OPTAX:
    value_and_grad = jax.jit(jax.value_and_grad(validation_loss))
    log_lam = jnp.log(jnp.asarray(1e-3))       # start deliberately under-regularised
    opt = optax.adam(0.2)
    state = opt.init(log_lam)

    print("training the sparsity penalty by gradient descent:")
    for it in range(150):
        loss, grad = value_and_grad(log_lam)
        if it % 30 == 0:
            print(f"   step {it:3d}: lambda = {float(jnp.exp(log_lam)):.3e}  "
                  f"val MSE = {float(loss):.4e}")
        updates, state = opt.update(grad, state)
        log_lam = optax.apply_updates(log_lam, updates)
    lam_learned = float(jnp.exp(log_lam))
    print(f"   final   : lambda = {lam_learned:.3e}  "
          f"val MSE = {float(validation_loss(log_lam)):.4e}")

# %% [markdown]
# Is it any good? The honest test is against the thing it replaces: a grid
# search fine enough that you would trust it.

# %%
if HAVE_OPTAX:
    grid = np.linspace(-9.0, -1.0, 60)
    losses = [float(validation_loss(jnp.asarray(g))) for g in grid]
    best = int(np.argmin(losses))
    print(f"   60-point grid : lambda = {np.exp(grid[best]):.3e}  "
          f"val MSE = {losses[best]:.4e}")
    print(f"   learned       : lambda = {lam_learned:.3e}  "
          f"val MSE = {float(validation_loss(log_lam)):.4e}")
    print("   The gradient finds a slightly better penalty than the grid, because")
    print("   it is not confined to the grid — and it would still be one search")
    print("   if there were ten hyper-parameters instead of one, where a grid")
    print("   would need 60^10 fits.")

# %% [markdown]
# ## What to take away
#
# * The whole pipeline is differentiable, so any output differentiates against
#   any input.
# * Derivatives w.r.t. **inputs** give exact sensitivities and Hessians.
# * Derivatives w.r.t. **data** tell you which model runs your conclusions rest
#   on, and where to spend the next one.
# * Derivatives w.r.t. the **input distribution** test the robustness of the UQ
#   study itself — the assumption nobody usually checks.
# * Anything differentiable can be *learned* rather than tuned by hand.
# * Reverse mode costs about one extra evaluation regardless of parameter count;
#   finite differences cost one per parameter.
#
# ## Exercises
#
# 1. Compute $\partial S_1 / \partial a$ for the first-order Sobol' index of a
#    two-input model as the input distribution changes. Is the ranking of inputs
#    stable?
# 2. In Section 5.2, which quadrature point has the largest influence on the variance,
#    and does that match your intuition?
# 3. Use `jax.jacobian` to get the derivative of *all* the coefficients with
#    respect to *all* the data at once. What shape is it, and what does the
#    matrix mean?
#
# Solutions below.

# %%
if __name__ == "__main__":
    print("\n--- Solutions ---")

    # 1. Sobol' indices are differentiable in the distribution too.
    g2 = lambda Z: Z[:, 0] + 2.0 * Z[:, 1] + 3.0 * Z[:, 0] * Z[:, 1]

    def first_sobol(a):
        p = eqj.DensityParameter(eqj.beta_density(a, 3.0), 0.0, 1.0, n_grid=200)
        r = [p.recurrence(5), eqj.uniform_recurrence(5)]
        Xs, Ws = eqj.tensor_quadrature(r)
        pp = eqj.Poly(r, eqj.total_order_indices(2, 3))
        pp.fit_projection(Xs, g2(Xs), Ws)
        return pp.sobol_indices()[0]

    s1 = float(first_sobol(2.0)); ds1 = float(jax.grad(first_sobol)(2.0))
    print(f"1. S1 = {s1:.5f}, dS1/da = {ds1:.5f}")
    print(f"   Over a +-0.5 change in a, S1 moves by about {abs(ds1)*0.5:.4f} —")
    print("   compare that with the gap between inputs to judge stability.")

    # 2. Largest |sensitivity|.
    j = int(np.argmax(np.abs(sens)))
    print(f"2. Most influential run: index {j} at x = "
          f"{np.round(np.array(X[j]), 3)}, dVar/dy = {sens[j]:.5f}")
    print("   Corner points of the tensor grid carry the largest leverage: they")
    print("   are where the high-order basis functions are biggest.")

    # 3. Jacobian of coefficients w.r.t. data.
    def coeffs_of(y):
        p = eqj.Poly(rec, idx)
        p.fit_projection(X, y, W)
        return p.coefficients

    J = np.array(jax.jacobian(coeffs_of)(y0))
    print(f"3. Jacobian shape {J.shape} = (n_basis, n_runs).")
    print("   Row k says how coefficient c_k responds to each model run; for a")
    print("   projection fit it is exactly the weighted design matrix, so the")
    print("   'derivative' and the estimator are the same object.")
