"""End-to-end tour of equadratures.jax (auto-differentiable equadratures).

Run:  python examples/jax_quickstart.py
Requires: pip install equadratures[jax-learn]
"""
import jax
import jax.numpy as jnp

import equadratures.jax as eqj


def demo_quadrature():
    alpha, beta, mu0 = eqj.legendre_recurrence(8)
    nodes, weights = eqj.gauss_quadrature(alpha, beta, mu0)
    approx = float(jnp.sum(weights * nodes ** 4))       # integral of x^4 on [-1,1]
    print("[quadrature] 8-pt Gauss-Legendre  int x^4 = %.10f  (exact 0.4)" % approx)


def demo_uq():
    rec = [eqj.uniform_recurrence(4), eqj.uniform_recurrence(4)]
    idx = eqj.total_order_indices(2, 2)
    X, W = eqj.tensor_quadrature(rec)
    f = lambda Z: 1.0 + 2.0 * Z[:, 0] + 3.0 * Z[:, 0] * Z[:, 1]
    poly = eqj.Poly(rec, idx)
    poly.fit_projection(X, f(X), W)
    print("[UQ] mean=%.6f  variance=%.6f  Sobol=%s  total-Sobol=%s"
          % (float(poly.mean()), float(poly.variance()),
             [round(float(s), 4) for s in poly.sobol_indices()],
             [round(float(t), 4) for t in poly.total_sobol_indices()]))
    # differentiate the surrogate and a UQ output
    dpred = jax.grad(lambda x: poly.predict(x[None, :])[0])(jnp.array([0.3, -0.4]))
    print("[UQ] d(surrogate)/dx at (0.3,-0.4) = %s  (analytic [0.8, 0.9])"
          % [round(float(g), 4) for g in dpred])


def demo_prescribed_nodes():
    alpha, beta, mu0 = eqj.legendre_recurrence(5)
    nodes, _ = eqj.radau_quadrature(alpha, beta, mu0, -1.0)
    print("[rules] 5-pt Radau  pins x=-1: %s" % [round(float(v), 4) for v in nodes])
    nodes, _ = eqj.lobatto_quadrature(alpha, beta, mu0, -1.0, 1.0)
    print("[rules] 5-pt Lobatto pins both ends: %s"
          % [round(float(v), 4) for v in nodes])


def demo_distribution_gradient():
    # Differentiate a UQ output w.r.t. the *input distribution's* shape.
    f = lambda Z: jnp.exp(Z[:, 0]) + Z[:, 0] ** 3

    def variance_given(a):
        p = eqj.DensityParameter(eqj.beta_density(a, 3.0), 0.0, 1.0, n_grid=250)
        rec = [p.recurrence(8)]
        X, W = eqj.tensor_quadrature(rec)
        poly = eqj.Poly(rec, eqj.total_order_indices(1, 5))
        poly.fit_projection(X, f(X), W)
        return poly.variance()

    print("[dist] Var[f(X)] with X~Beta(2,3) = %.6f" % float(variance_given(2.0)))
    print("[dist] d(Var)/d(Beta shape a)    = %.6f  <- the classic namespace "
          "cannot produce this" % float(jax.grad(variance_given)(2.0)))


def demo_sparse():
    # 20 samples, 28 basis terms: least squares is under-determined, l1 is not.
    import numpy as np
    rng = np.random.default_rng(7)
    rec = [eqj.uniform_recurrence(8), eqj.uniform_recurrence(8)]
    idx = eqj.total_order_indices(2, 6)
    X = jnp.asarray(rng.uniform(-1, 1, size=(20, 2)))
    f = lambda Z: 1.0 + 2.0 * Z[:, 0] + 3.0 * Z[:, 0] * Z[:, 1]

    poly = eqj.Poly(rec, idx)
    c = poly.fit_lasso(X, f(X), 1e-3, max_iter=8000)
    Xt = jnp.asarray(rng.uniform(-1, 1, size=(15, 2)))
    err = float(jnp.max(jnp.abs(poly.predict(Xt) - f(Xt))))
    print("[sparse] %d of %d basis terms active; max test error %.2e"
          % (int(jnp.count_nonzero(c)), idx.shape[0], err))

    # the penalty is a differentiable input, so it can be trained
    def val_loss(l1):
        p = eqj.Poly(rec, idx)
        p.fit_elastic_net(X, f(X), l1, 1e-6, max_iter=4000)
        return jnp.mean((p.predict(Xt) - f(Xt)) ** 2)

    print("[sparse] d(validation MSE)/d(penalty) = %.4e"
          % float(jax.grad(val_loss)(1e-3)))


def demo_learnable_kernel():
    try:
        import optax
    except ImportError:
        print("[kernel] optax not installed; skipping (pip install equadratures[jax-learn])")
        return
    p = 10
    kernel = eqj.PolynomialKernel([eqj.uniform_recurrence(p + 1)],
                                  eqj.total_order_indices(1, p))
    Xtr = jnp.linspace(-1, 1, 20).reshape(-1, 1)
    Xte = jnp.linspace(-0.95, 0.95, 50).reshape(-1, 1)
    g = lambda X: jnp.sin(3.0 * X[:, 0])
    ytr, yte = g(Xtr), g(Xte)

    params = {"log_theta": kernel.default_log_theta(), "log_noise": jnp.log(1e-2)}
    loss = lambda pr: eqj.gp_nlml(kernel, Xtr, ytr, pr["log_theta"], pr["log_noise"])
    mse = lambda pr: float(jnp.mean(
        (eqj.gp_predict(kernel, Xtr, ytr, Xte, pr["log_theta"], pr["log_noise"]) - yte) ** 2))

    nlml0, mse0 = float(loss(params)), mse(params)
    opt = optax.adam(5e-2)
    state = opt.init(params)
    step = jax.jit(jax.value_and_grad(loss))
    for _ in range(300):
        _, grads = step(params)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    print("[kernel] learnable GP:  NLML %.3f -> %.3f   test-MSE %.2e -> %.2e"
          % (nlml0, float(loss(params)), mse0, mse(params)))


if __name__ == "__main__":
    demo_quadrature()
    demo_prescribed_nodes()
    demo_uq()
    demo_distribution_gradient()
    demo_sparse()
    demo_learnable_kernel()
