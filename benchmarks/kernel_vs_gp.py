"""Polynomial (Mercer) kernel versus a standard RBF/Matern GP — deliverable D3.

The grant argues that kernels grounded in polynomial approximation theory are
*competitive* with the usual stationary kernels and *more interpretable*. That is
a testable claim, so this script tests it rather than asserting it.

Both models are implemented on the same JAX code path — same marginal
likelihood, same Cholesky solve, same Optax training loop — so the comparison
measures the kernels, not two libraries' engineering. No scikit-learn dependency
is introduced.

Compared on four axes:

1. **Accuracy** — RMSE on held-out points.
2. **Calibration** — do the error bars mean anything? Measured by 90% interval
   coverage (want ~0.90) and mean negative log predictive density (lower is
   better). A model that is accurate but overconfident is not a good GP.
3. **Interpretability** — what the fitted hyper-parameters actually tell you.
4. **Cost** — training and prediction time.

Test problems are chosen to include a case each kernel should *lose*, because a
benchmark that its author always wins is not evidence:

* ``smooth``    — analytic, entire. Polynomials should excel.
* ``oscillatory`` — high-frequency; needs many polynomial modes.
* ``kink``      — continuous but not differentiable. Polynomials should struggle
                  (Gibbs-type behaviour); a Matern kernel is built for this.
* ``smooth_2d`` — two inputs, to check the story survives past one dimension.

Usage::

    python benchmarks/kernel_vs_gp.py            # full run
    python benchmarks/kernel_vs_gp.py --quick    # fewer steps, for CI
    python benchmarks/kernel_vs_gp.py --out results.md

Convention note: ``log_noise`` is the log of the noise **variance**, matching
:func:`equadratures.jax.kernel.gp_nlml`.
"""
import argparse
import platform
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

import equadratures.jax as eqj


# --------------------------------------------------------------- GP machinery
def rbf_gram(X1, X2, params):
    """Squared-exponential kernel with a learnable length scale and amplitude."""
    ell = jnp.exp(params["log_ell"])
    amp = jnp.exp(2.0 * params["log_amp"])
    d2 = jnp.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    return amp * jnp.exp(-0.5 * d2 / ell ** 2)


def matern52_gram(X1, X2, params):
    """Matern 5/2 — the standard choice when the target is not very smooth."""
    ell = jnp.exp(params["log_ell"])
    amp = jnp.exp(2.0 * params["log_amp"])
    d2 = jnp.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    r = jnp.sqrt(jnp.maximum(d2, 1e-30)) / ell
    s5 = jnp.sqrt(5.0) * r
    return amp * (1.0 + s5 + 5.0 * r ** 2 / 3.0) * jnp.exp(-s5)


def poly_gram_factory(kernel):
    def gram(X1, X2, params):
        return kernel.gram(X1, X2, params["log_theta"])
    return gram


def nlml(gram, X, y, params):
    """Negative log marginal likelihood, shared by every kernel here."""
    m = X.shape[0]
    K = gram(X, X, params) + jnp.exp(params["log_noise"]) * jnp.eye(m)
    L = jnp.linalg.cholesky(K)
    a = jsl.solve_triangular(L, y, lower=True)
    return 0.5 * jnp.dot(a, a) + jnp.sum(jnp.log(jnp.diag(L))) \
        + 0.5 * m * jnp.log(2.0 * jnp.pi)


def predict(gram, Xtr, ytr, Xte, params):
    """Posterior mean and *predictive* variance (latent + observation noise)."""
    m = Xtr.shape[0]
    noise = jnp.exp(params["log_noise"])
    K = gram(Xtr, Xtr, params) + noise * jnp.eye(m)
    L = jnp.linalg.cholesky(K)
    alpha = jsl.solve_triangular(L.T, jsl.solve_triangular(L, ytr, lower=True),
                                 lower=False)
    Ks = gram(Xte, Xtr, params)
    mean = Ks @ alpha
    v = jsl.solve_triangular(L, Ks.T, lower=True)
    k_diag = jnp.diag(gram(Xte, Xte, params))
    var = jnp.maximum(k_diag - jnp.sum(v ** 2, axis=0), 1e-12) + noise
    return mean, var


def train(gram, Xtr, ytr, params, steps, lr=5e-2):
    """Fit hyper-parameters by Adam on the marginal likelihood."""
    import optax

    opt = optax.adam(lr)
    state = opt.init(params)
    step = jax.jit(jax.value_and_grad(lambda p: nlml(gram, Xtr, ytr, p)))
    first = float(step(params)[0])
    for _ in range(steps):
        _, grads = step(params)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    return params, first, float(nlml(gram, Xtr, ytr, params))


# ------------------------------------------------------------------ problems
def make_problem(name, rng, n_train=30, n_test=400):
    """Return ``(Xtr, ytr, Xte, yte, dim, noise_sd)``."""
    if name == "smooth":
        f = lambda X: np.sin(3.0 * X[:, 0]) * np.exp(-X[:, 0])
        dim = 1
    elif name == "oscillatory":
        f = lambda X: np.sin(9.0 * X[:, 0]) + 0.3 * X[:, 0]
        dim = 1
    elif name == "kink":
        f = lambda X: np.abs(X[:, 0] - 0.15)
        dim = 1
    elif name == "smooth_2d":
        f = lambda X: np.exp(-X[:, 0]) * np.sin(2.0 * X[:, 1]) + 0.5 * X[:, 0] * X[:, 1]
        dim = 2
    else:                                             # pragma: no cover
        raise ValueError(name)

    noise_sd = 0.02
    Xtr = rng.uniform(-1, 1, size=(n_train, dim))
    Xte = rng.uniform(-1, 1, size=(n_test, dim))
    ytr = f(Xtr) + noise_sd * rng.normal(size=n_train)
    return (jnp.asarray(Xtr), jnp.asarray(ytr),
            jnp.asarray(Xte), jnp.asarray(f(Xte)), dim, noise_sd)


# ------------------------------------------------------------------- metrics
def evaluate(mean, var, yte):
    mean, var, yte = np.array(mean), np.array(var), np.array(yte)
    sd = np.sqrt(var)
    rmse = float(np.sqrt(np.mean((mean - yte) ** 2)))
    z = 1.6448536269514722                            # 90% two-sided normal
    coverage = float(np.mean(np.abs(mean - yte) <= z * sd))
    nlpd = float(np.mean(0.5 * np.log(2 * np.pi * var)
                         + 0.5 * (yte - mean) ** 2 / var))
    return rmse, coverage, nlpd


def _fmt(x):
    return "%.3e" % x if abs(x) < 1e-2 else "%.4f" % x


# --------------------------------------------------------------------- driver
def run(problem, quick, seed=0):
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte, dim, noise_sd = make_problem(problem, rng)
    steps = 200 if quick else 600
    order = 8 if dim == 1 else 5

    rec = [eqj.uniform_recurrence(order + 1)] * dim
    idx = eqj.total_order_indices(dim, order)
    poly_kernel = eqj.PolynomialKernel(rec, idx)

    log_noise0 = float(np.log(noise_sd ** 2))
    models = {
        "polynomial": (poly_gram_factory(poly_kernel),
                       {"log_theta": poly_kernel.default_log_theta(),
                        "log_noise": jnp.asarray(log_noise0)}),
        "RBF": (rbf_gram,
                {"log_ell": jnp.asarray(0.0), "log_amp": jnp.asarray(0.0),
                 "log_noise": jnp.asarray(log_noise0)}),
        "Matern 5/2": (matern52_gram,
                       {"log_ell": jnp.asarray(0.0), "log_amp": jnp.asarray(0.0),
                        "log_noise": jnp.asarray(log_noise0)}),
    }

    rows = []
    for name, (gram, p0) in models.items():
        t0 = time.perf_counter()
        params, nlml0, nlml1 = train(gram, Xtr, ytr, p0, steps)
        t_train = time.perf_counter() - t0

        pred_fn = jax.jit(lambda p: predict(gram, Xtr, ytr, Xte, p))
        m, v = pred_fn(params)
        m.block_until_ready()
        t0 = time.perf_counter()
        m, v = pred_fn(params)
        m.block_until_ready()
        t_pred = time.perf_counter() - t0

        rmse, cov, nlpd = evaluate(m, v, yte)
        rows.append((name, rmse, cov, nlpd, nlml0, nlml1, t_train, t_pred,
                     params))
    return rows, poly_kernel


def describe_hypers(name, params, kernel):
    """The interpretability axis, stated concretely rather than asserted."""
    if name == "polynomial":
        w = np.array(jnp.exp(2.0 * params["log_theta"]))
        active = int((w > 0.01 * w.max()).sum())
        top = sorted(int(k) for k in np.argsort(-w)[:3])
        return (f"spectrum over {len(w)} polynomial modes; {active} carry >1% of "
                f"the peak weight, strongest degrees {top} — reads directly as "
                f"'how much structure of each order the data support'")
    ell = float(jnp.exp(params["log_ell"]))
    amp = float(jnp.exp(params["log_amp"]))
    return (f"length scale {ell:.3f}, amplitude {amp:.3f} — two numbers, and "
            f"neither says which structure was used")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    try:
        import optax                                   # noqa: F401
    except ImportError:                                # pragma: no cover
        print("optax is required: pip install equadratures[jax-learn]")
        return 1

    out = [
        "# Polynomial kernel versus standard GP kernels",
        "",
        "- platform: `%s`, Python %s" % (platform.platform(),
                                         platform.python_version()),
        "- jax %s on %s" % (jax.__version__,
                            ", ".join(str(d.device_kind) for d in jax.devices())),
        "- mode: %s" % ("quick" if args.quick else "full"),
        "",
        "All kernels share one implementation of the marginal likelihood, the "
        "Cholesky solve and the Adam training loop, so what is compared is the "
        "kernel and nothing else. `coverage` is the fraction of held-out points "
        "inside the 90% predictive interval — **0.90 is the target, and higher "
        "is not better**. `NLPD` is the mean negative log predictive density; "
        "lower is better and it rewards being both accurate *and* honest.",
        "",
    ]

    for problem in ("smooth", "oscillatory", "kink", "smooth_2d"):
        rows, kernel = run(problem, args.quick)
        out += ["## `%s`" % problem, "",
                "| kernel | RMSE | 90% coverage | NLPD | NLML before → after | train (s) | predict |",
                "|---|---|---|---|---|---|---|"]
        for (name, rmse, cov, nlpd, n0, n1, tt, tp, _p) in rows:
            out.append("| %s | %s | %.2f | %.3f | %.1f → %.1f | %.2f | %.2f ms |"
                       % (name, _fmt(rmse), cov, nlpd, n0, n1, tt, tp * 1e3))
        best = min(rows, key=lambda r: r[1])[0]
        out += ["", "Lowest RMSE: **%s**." % best, "", "Hyper-parameters after training:", ""]
        for (name, *_rest, params) in rows:
            out.append("- **%s** — %s" % (name, describe_hypers(name, params, kernel)))
        out.append("")

    out += [
        "## Reading these numbers",
        "",
        "The polynomial kernel is expected to win on `smooth` and to lose on "
        "`kink`: a kink is exactly the case stationary kernels of low "
        "smoothness were designed for, and no polynomial basis represents it "
        "efficiently. Reporting that loss is the point — a kernel that claimed "
        "to win everywhere would not be believable.",
        "",
        "The durable advantage is the last column of prose rather than the "
        "table: an RBF fit yields a length scale, whereas the polynomial fit "
        "yields a spectrum over interpretable modes, which is a statement about "
        "the *function* and not merely about the fit.",
        "",
    ]

    text = "\n".join(out)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
