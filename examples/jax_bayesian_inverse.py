"""Bayesian inverse problems on a differentiable equadratures surrogate.

Two demonstrations of what auto-differentiation buys you downstream:

  1. inverse UQ  -- recover unknown model inputs from noisy measurements, with
     NUTS running straight through the polynomial surrogate;
  2. Bayesian compressed sensing -- infer a sparse polynomial expansion with
     credible intervals, the probabilistic counterpart of the l1 solve.

Both need gradients of the forward model, which is precisely what the classic
NumPy namespace cannot supply.

Run:  python examples/jax_bayesian_inverse.py
Requires: pip install equadratures[jax-bayes]
"""
import numpy as np
import jax
import jax.numpy as jnp

import equadratures.jax as eqj
from equadratures.jax import inverse as eqi


def demo_inverse_uq():
    """Infer two unknown inputs from seven noisy measurements."""
    # --- build a surrogate of the (notionally expensive) forward model -------
    # Variables: x1, x2 are unknown inputs; t is a control the rig sweeps.
    def truth(Z):
        x1, x2, t = Z[:, 0], Z[:, 1], Z[:, 2]
        return 1.0 + 2.0 * x1 + 0.5 * x1 ** 2 + x2 * t + 0.3 * x1 * t ** 2

    order = 3
    rec = [eqj.uniform_recurrence(order + 2)] * 3
    idx = eqj.total_order_indices(3, order)
    X, W = eqj.tensor_quadrature(rec)
    poly = eqj.Poly(rec, idx)
    poly.fit_projection(X, truth(X), W)

    # --- synthesise measurements at a known truth ---------------------------
    t_grid = jnp.linspace(-0.9, 0.9, 7)
    x_true = jnp.asarray([0.3, -0.5])
    noise = 0.02

    def forward(x):
        Z = jnp.stack([jnp.full_like(t_grid, x[0]),
                       jnp.full_like(t_grid, x[1]), t_grid], axis=1)
        return poly.predict(Z)

    key = jax.random.PRNGKey(0)
    y_obs = forward(x_true) + noise * jax.random.normal(key, t_grid.shape)

    # the gradient NUTS will use at every leapfrog step
    g = jax.grad(lambda x: jnp.sum(forward(x) ** 2))(x_true)
    print("[inverse] surrogate is differentiable: d/dx sum(f^2) = %s"
          % [round(float(v), 4) for v in g])

    # --- sample the posterior over the inputs -------------------------------
    model = eqi.surrogate_input_model(forward, y_obs, lower=[-1.0, -1.0],
                                      upper=[1.0, 1.0], noise_scale=noise)
    mcmc = eqi.run_nuts(model, num_warmup=500, num_samples=1500, seed=0)
    x = np.array(eqi.posterior_samples(mcmc, "x"))

    mean = x.mean(axis=0)
    lo, hi = np.quantile(x, 0.05, axis=0), np.quantile(x, 0.95, axis=0)
    print("[inverse] truth      = %s" % [round(float(v), 4) for v in x_true])
    print("[inverse] posterior  = %s  (prior was uniform on [-1, 1])"
          % [round(float(v), 4) for v in mean])
    for i in range(2):
        print("[inverse]   x%d 90%% CI = [%.4f, %.4f]  contains truth: %s"
              % (i + 1, lo[i], hi[i], bool(lo[i] <= x_true[i] <= hi[i])))

    try:
        import arviz as az
        rhat = az.rhat(eqi.to_arviz(
            eqi.run_nuts(model, num_warmup=300, num_samples=500,
                         num_chains=2, seed=1)))["x"].values
        print("[inverse] ArviZ r_hat = %s  (converged if < 1.01)"
              % [round(float(v), 4) for v in np.atleast_1d(rhat)])
    except ImportError:
        print("[inverse] arviz not installed; skipping diagnostics")


def demo_bayesian_compressed_sensing():
    """Sparse polynomial recovery, with uncertainty on every coefficient."""
    rng = np.random.default_rng(3)
    order = 4
    rec = [eqj.uniform_recurrence(order + 2)] * 2
    idx = eqj.total_order_indices(2, order)              # 15 basis terms
    Xs = jnp.asarray(rng.uniform(-1.0, 1.0, size=(25, 2)))
    A = eqj.design_matrix(Xs, idx, rec)

    c_true = np.zeros(idx.shape[0])
    support = [0, 2, 4]
    c_true[support] = [1.5, -2.0, 1.0]
    y = A @ jnp.asarray(c_true)

    # point estimate from the l1 solve ...
    c_l1 = np.array(eqj.lasso(A, y, 1e-3, max_iter=8000))
    # ... and the full posterior under a horseshoe prior
    model = eqi.sparse_coefficient_model(A, y, tau_scale=0.5, noise_scale=0.01)
    mcmc = eqi.run_nuts(model, num_warmup=600, num_samples=1000, seed=0,
                        target_accept_prob=0.9)
    summary = eqi.summarise(mcmc)
    mean = np.array(summary["c"]["mean"])
    std = np.array(summary["c"]["std"])

    print("\n[sparse] term |    truth |    l1 |  posterior mean +- sd")
    for k in range(idx.shape[0]):
        flag = " *" if k in support else "  "
        print("[sparse]%s %3d | %8.4f | %7.4f | %8.4f +- %.4f"
              % (flag, k, c_true[k], c_l1[k], mean[k], std[k]))
    off = [k for k in range(idx.shape[0]) if k not in support]
    print("[sparse] active terms marked *; largest inactive posterior mean "
          "= %.4f" % np.abs(mean[off]).max())


if __name__ == "__main__":
    demo_inverse_uq()
    demo_bayesian_compressed_sensing()
