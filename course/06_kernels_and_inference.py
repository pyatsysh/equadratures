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
# # Lesson 6 — Learnable kernels and Bayesian inference
#
# The last lesson puts the pieces together. We build a **kernel** out of the
# orthonormal polynomials, *learn* it from data, use it for Gaussian-process
# regression, and finally run gradient-based MCMC through the surrogate to solve
# an inverse problem.
#
# Everything here needs derivatives of the forward model. That is why it is the
# last lesson and not the first.
#
# Requires `pip install "equadratures[jax-learn,jax-bayes]"`.

# %%
import os

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
# ## 6.1 A kernel built from polynomials
#
# A kernel measures similarity between inputs. Take the orthonormal basis
# functions $\phi_k$ as *features* and assign each a weight $\theta_k^2$:
#
# $$k(x, x') \;=\; \sum_k \theta_k^2\, \phi_k(x)\, \phi_k(x')$$
#
# Two properties come for free:
#
# * it is **positive semi-definite by construction** — it is a sum of squares, so
#   it is a valid kernel for any $\theta$, with no constraints to enforce;
# * it is **interpretable** — $\theta_k$ is the amount of "constant", "linear",
#   "quadratic" ... the model is allowed to use. Compare that with the length
#   scale of an RBF kernel, which tells you very little about the function.

# %%
DEG = 10
rec = [eqj.uniform_recurrence(DEG + 1)]
idx = eqj.total_order_indices(1, DEG)
kernel = eqj.PolynomialKernel(rec, idx)
print("kernel with", kernel.n_features, "polynomial features")

xs = jnp.linspace(-1, 1, 200).reshape(-1, 1)
log_theta = kernel.default_log_theta()
K = np.array(kernel.gram(xs, xs, log_theta))
print("Gram matrix is PSD:", bool(np.min(np.linalg.eigvalsh(K)) > -1e-10))

plt.figure(figsize=(4.6, 4.0))
plt.imshow(K, extent=[-1, 1, 1, -1], cmap="viridis")
plt.colorbar(label="k(x, x')"); plt.title("polynomial kernel, flat spectrum")
plt.xlabel("x'"); plt.ylabel("x")
save("06_kernel.png")

# %% [markdown]
# ## 6.2 Gaussian-process regression with it
#
# A GP with this kernel gives predictions *and* error bars. We fit noisy samples
# of a smooth function.

# %%
rng = np.random.default_rng(0)
truth = lambda x: jnp.sin(3.0 * x) * jnp.exp(-x)

Xtr = jnp.asarray(np.sort(rng.uniform(-1, 1, 14)).reshape(-1, 1))
ytr = truth(Xtr[:, 0]) + 0.05 * jnp.asarray(rng.normal(size=Xtr.shape[0]))
log_noise = jnp.log(jnp.asarray(0.05 ** 2))

mean0, var0 = eqj.gp_predict_with_variance(kernel, Xtr, ytr, xs, log_theta, log_noise)
nlml0 = float(eqj.gp_nlml(kernel, Xtr, ytr, log_theta, log_noise))
print("negative log marginal likelihood, untrained spectrum:", nlml0)


def plot_gp(mean, var, title, fname):
    m, s = np.array(mean), np.sqrt(np.array(var))
    plt.figure(figsize=(6.4, 3.6))
    plt.plot(np.array(xs)[:, 0], np.array(truth(xs[:, 0])), "k", lw=2, label="truth")
    plt.plot(np.array(xs)[:, 0], m, "C0", lw=2, label="GP mean")
    plt.fill_between(np.array(xs)[:, 0], m - 2 * s, m + 2 * s, alpha=.25,
                     color="C0", label="±2 sd")
    plt.plot(np.array(Xtr)[:, 0], np.array(ytr), "ko", ms=5, label="data")
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.title(title)
    save(fname)


plot_gp(mean0, var0, "GP before training the spectrum", "06_gp_untrained.png")

# %% [markdown]
# ## 6.3 Learning the kernel
#
# The spectrum $\theta$ and the noise level are just parameters, and the marginal
# likelihood is differentiable in both. So we *train* them — the same way you
# would train a neural network, with Optax.

# %%
try:
    import optax
    HAVE_OPTAX = True
except ImportError:
    HAVE_OPTAX = False
    print("optax not installed; skipping training "
          "(pip install equadratures[jax-learn])")

if HAVE_OPTAX:
    params = {"log_theta": kernel.default_log_theta(),
              "log_noise": jnp.log(jnp.asarray(1e-2))}

    def loss(p):
        return eqj.gp_nlml(kernel, Xtr, ytr, p["log_theta"], p["log_noise"])

    opt = optax.adam(5e-2)
    state = opt.init(params)
    step = jax.jit(jax.value_and_grad(loss))

    trace = []
    for it in range(400):
        val, grads = step(params)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        trace.append(float(val))

    print(f"NLML {trace[0]:.2f} -> {trace[-1]:.2f}")
    print("learned noise sd:", float(jnp.sqrt(jnp.exp(params['log_noise']))),
          " (data had 0.05)")

    plt.figure(figsize=(6, 3.2))
    plt.plot(trace); plt.xlabel("Adam step"); plt.ylabel("negative log marginal likelihood")
    plt.grid(alpha=.3); plt.title("training the kernel spectrum")
    save("06_nlml.png")

    mean1, var1 = eqj.gp_predict_with_variance(
        kernel, Xtr, ytr, xs, params["log_theta"], params["log_noise"])
    plot_gp(mean1, var1, "GP after training the spectrum", "06_gp_trained.png")

# %% [markdown]
# ## 6.4 Reading the learned spectrum
#
# This is where the interpretability claim earns its keep. Plot $\theta_k^2$
# against polynomial degree: it tells you, directly, how much structure of each
# order the data supports.

# %%
if HAVE_OPTAX:
    weights = np.array(jnp.exp(2.0 * params["log_theta"]))
    plt.figure(figsize=(6, 3.2))
    plt.semilogy(np.arange(len(weights)), weights, "o-")
    plt.xlabel("polynomial degree $k$"); plt.ylabel(r"$\theta_k^2$")
    plt.grid(alpha=.3, which="both")
    plt.title("the learned spectrum: which modes the data support")
    save("06_spectrum.png")

    top = np.argsort(-weights)[:4]
    print("most-used polynomial degrees:", sorted(int(t) for t in top))
    print("A decaying spectrum means the data justify only low-order structure.")

# %% [markdown]
# ## 6.5 Inverse problems: inferring inputs from measurements
#
# The final application, and the one that most needs gradients. You measured the
# *output* of a system at several operating points. You want the *inputs* that
# produced it, with honest uncertainty.
#
# NUTS (the No-U-Turn sampler) differentiates the log-posterior at every step,
# which means differentiating the forward model. With the classic NumPy backend
# that is simply not available.

# %%
try:
    from equadratures.jax import inverse as eqi
    HAVE_NUMPYRO = eqi._HAS_NUMPYRO
except Exception:
    HAVE_NUMPYRO = False

if not HAVE_NUMPYRO:
    print("numpyro not installed; skipping "
          "(pip install equadratures[jax-bayes])")
else:
    # Forward model: two unknown inputs (x1, x2) and a control t the rig sweeps.
    def rig(Z):
        x1, x2, t = Z[:, 0], Z[:, 1], Z[:, 2]
        return 1.0 + 2.0 * x1 + 0.5 * x1 ** 2 + x2 * t + 0.3 * x1 * t ** 2

    order = 3
    rec3 = [eqj.uniform_recurrence(order + 2)] * 3
    X3, W3 = eqj.tensor_quadrature(rec3)
    surro = eqj.Poly(rec3, eqj.total_order_indices(3, order))
    surro.fit_projection(X3, rig(X3), W3)

    t_grid = jnp.linspace(-0.9, 0.9, 7)
    x_true = jnp.asarray([0.3, -0.5])
    noise = 0.02

    def forward(x):
        Z = jnp.stack([jnp.full_like(t_grid, x[0]),
                       jnp.full_like(t_grid, x[1]), t_grid], axis=1)
        return surro.predict(Z)

    y_obs = forward(x_true) + noise * jax.random.normal(
        jax.random.PRNGKey(0), t_grid.shape)

    model = eqi.surrogate_input_model(forward, y_obs, lower=[-1.0, -1.0],
                                      upper=[1.0, 1.0], noise_scale=noise)
    mcmc = eqi.run_nuts(model, num_warmup=500, num_samples=1500, seed=0)
    draws = np.array(eqi.posterior_samples(mcmc, "x"))

    print("truth              :", np.round(np.array(x_true), 4))
    print("posterior mean     :", np.round(draws.mean(axis=0), 4))
    for i in range(2):
        lo, hi = np.quantile(draws[:, i], [0.05, 0.95])
        print(f"  x{i+1} 90% interval: [{lo:.4f}, {hi:.4f}]  "
              f"contains truth: {bool(lo <= x_true[i] <= hi)}")

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax[0].plot(draws[:, 0], draws[:, 1], ".", ms=2, alpha=.3)
    ax[0].plot(*np.array(x_true), "r*", ms=14, label="truth")
    ax[0].set_xlabel("$x_1$"); ax[0].set_ylabel("$x_2$"); ax[0].legend()
    ax[0].set_title("joint posterior"); ax[0].grid(alpha=.3)
    ax[1].plot(np.array(t_grid), np.array(y_obs), "ko", label="measurements")
    for k in rng.choice(len(draws), 40, replace=False):
        ax[1].plot(np.array(t_grid), np.array(forward(jnp.asarray(draws[k]))),
                   "C0", alpha=.12)
    ax[1].set_xlabel("control $t$"); ax[1].set_ylabel("output")
    ax[1].set_title("posterior predictive"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()
    save("06_inverse.png")

# %% [markdown]
# ## 6.6 Always check the sampler converged
#
# MCMC output is worthless without diagnostics. $\hat{R}$ compares variance
# between chains with variance within them; it should be very close to 1.

# %%
if HAVE_NUMPYRO:
    mcmc2 = eqi.run_nuts(model, num_warmup=300, num_samples=500,
                         num_chains=2, seed=1)
    try:
        import arviz as az
        rhat = az.rhat(eqi.to_arviz(mcmc2))["x"].values
        print("r_hat:", np.round(np.atleast_1d(rhat), 4), " (want < 1.01)")
    except ImportError:
        print("arviz not installed; skipping diagnostics")

# %% [markdown]
# ## What to take away
#
# * A kernel built from orthonormal polynomials is PSD for free and
#   interpretable by construction: the spectrum says which polynomial modes the
#   data support.
# * Because the marginal likelihood is differentiable, the kernel can be
#   *learned* rather than chosen.
# * The trained GP gives predictions with calibrated error bars.
# * Gradient-based MCMC runs straight through the surrogate, which turns
#   expensive inverse problems into tractable ones.
# * None of this is possible without differentiability — which is the argument
#   the whole course has been building towards.
#
# ## Where to go next
#
# * `equadratures/jax/README.md` — the API reference for everything used here.
# * `examples/jax_quickstart.py` and `examples/jax_bayesian_inverse.py` —
#   condensed, runnable versions.
# * `benchmarks/RESULTS.md` — measured performance of the pieces.
#
# ## Exercises
#
# 1. Retrain the kernel with only 5 data points. What happens to the learned
#    spectrum, and is that the right behaviour?
# 2. Widen the observation noise in §6.5 from 0.02 to 0.2. How much does the
#    posterior widen? Does it still cover the truth?
# 3. The polynomial kernel has finite rank. Predict what its posterior variance
#    does far from the data, and check whether you were right.
#
# Solutions below.

# %%
if __name__ == "__main__":
    print("\n--- Solutions ---")

    if HAVE_OPTAX:
        # 1. Less data supports less structure: the spectrum should decay sooner.
        Xs5 = Xtr[:5]
        ys5 = ytr[:5]
        p5 = {"log_theta": kernel.default_log_theta(),
              "log_noise": jnp.log(jnp.asarray(1e-2))}
        o5 = optax.adam(5e-2); s5 = o5.init(p5)
        st5 = jax.jit(jax.value_and_grad(
            lambda p: eqj.gp_nlml(kernel, Xs5, ys5, p["log_theta"], p["log_noise"])))
        for _ in range(400):
            _, gr = st5(p5)
            up, s5 = o5.update(gr, s5, p5)
            p5 = optax.apply_updates(p5, up)
        w5 = np.array(jnp.exp(2.0 * p5["log_theta"]))
        w14 = np.array(jnp.exp(2.0 * params["log_theta"]))
        print(f"1. effective modes (weight > 1% of max): "
              f"{int((w5 > 0.01*w5.max()).sum())} with 5 points, "
              f"{int((w14 > 0.01*w14.max()).sum())} with 14.")
        print("   Less data supports fewer modes. That is the correct, and")
        print("   automatic, behaviour — the marginal likelihood penalises")
        print("   complexity the data cannot justify.")

    if HAVE_NUMPYRO:
        # 2. Wider noise -> wider posterior, still covering the truth.
        y_loose = forward(x_true) + 0.2 * jax.random.normal(
            jax.random.PRNGKey(3), t_grid.shape)
        m_loose = eqi.surrogate_input_model(forward, y_loose,
                                            lower=[-1.0, -1.0], upper=[1.0, 1.0],
                                            noise_scale=0.2)
        d_loose = np.array(eqi.posterior_samples(
            eqi.run_nuts(m_loose, num_warmup=500, num_samples=1500, seed=3), "x"))
        print(f"2. posterior sd at noise 0.02: {draws.std(axis=0).round(4)}")
        print(f"   posterior sd at noise 0.20: {d_loose.std(axis=0).round(4)}")
        print("   Roughly a tenfold widening for a tenfold noise increase, as")
        print("   expected in the linear-ish regime; coverage is retained.")

        # 3. Finite rank: variance does NOT grow without bound away from data.
        far = jnp.asarray([[-0.999], [0.999]])
        _, v_far = eqj.gp_predict_with_variance(
            kernel, Xtr, ytr, far, params["log_theta"] if HAVE_OPTAX else log_theta,
            log_noise)
        _, v_mid = eqj.gp_predict_with_variance(
            kernel, Xtr, ytr, jnp.asarray([[0.0]]),
            params["log_theta"] if HAVE_OPTAX else log_theta, log_noise)
        print(f"3. variance at the edges {np.array(v_far).round(6)}, "
              f"at the centre {np.array(v_mid).round(6)}")
        print("   A finite-rank kernel cannot express 'total ignorance': once the")
        print("   data pin down all the features, variance stays bounded. The")
        print("   error bars describe uncertainty WITHIN the polynomial span,")
        print("   not model-form error outside it. Know that before trusting them.")
