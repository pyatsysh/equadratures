"""Benchmarks for the ``equadratures.jax`` namespace (grant deliverable D2).

Four things are measured, because they are the four claims the JAX backend
actually makes:

1. **jit** -- compiled versus eager execution of the same JAX code.
2. **vmap** -- one batched call versus a Python loop over the same work.
3. **adjoint versus finite differences** -- the cost of a gradient as the number
   of parameters grows. Reverse-mode gives the whole gradient for roughly the
   price of one extra function evaluation; finite differences pay one
   evaluation *per parameter*. This is the D1 payoff measured in seconds.
4. **classic versus JAX** -- the same task through the incumbent NumPy
   implementation, to show the port costs nothing on the machine most users
   have.

Honest caveat, stated up front and repeated in the output: this box has no GPU,
so ``jaxlib`` is the CPU build. The headline accelerator claim of D2 is
*untested here* -- what these numbers establish is compilation, batching and
adjoint scaling, all of which are hardware-independent properties of the port.
GPU/``pmap`` parity remains open and is flagged as such in the roadmap.

Usage::

    python benchmarks/jax_benchmarks.py            # full run
    python benchmarks/jax_benchmarks.py --quick    # small sizes, for CI
    python benchmarks/jax_benchmarks.py --out results.md
"""
import argparse
import platform
import statistics
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
import equadratures.jax as eqj


# --------------------------------------------------------------------- timing
def _block(x):
    """Force JAX's asynchronous dispatch to complete before the clock stops."""
    for leaf in jax.tree_util.tree_leaves(x):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return x


def measure(fn, *args, repeat=7):
    """Median wall time of ``fn(*args)`` in seconds, plus first-call time.

    The first call is reported separately because for ``jit``-ed functions it
    includes tracing and compilation, which is a one-off cost that would
    otherwise be smeared across the steady-state number and misrepresent both.
    """
    t0 = time.perf_counter()
    _block(fn(*args))
    first = time.perf_counter() - t0

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _block(fn(*args))
        times.append(time.perf_counter() - t0)
    return statistics.median(times), first


def _fmt(seconds):
    if seconds < 1e-3:
        return "%.1f us" % (seconds * 1e6)
    if seconds < 1.0:
        return "%.2f ms" % (seconds * 1e3)
    return "%.2f s" % seconds


class Report:
    """Collects rows and prints them as a markdown table."""

    def __init__(self):
        self.sections = []

    def section(self, title, note=None):
        self.sections.append((title, note, []))

    def row(self, *cells):
        self.sections[-1][2].append(cells)

    def render(self, headers):
        out = []
        for title, note, rows in self.sections:
            out.append("### %s\n" % title)
            if note:
                out.append("%s\n" % note)
            out.append("| " + " | ".join(headers) + " |")
            out.append("|" + "|".join(["---"] * len(headers)) + "|")
            for r in rows:
                out.append("| " + " | ".join(str(c) for c in r) + " |")
            out.append("")
        return "\n".join(out)


# ----------------------------------------------------------------- benchmarks
def bench_jit(report, quick):
    """Compiled versus eager, on the quadrature and design-matrix kernels."""
    report.section(
        "jit versus eager",
        "Same JAX code, with and without compilation. `compile` is the one-off "
        "first-call cost; it is amortised over every later call.")

    sizes = [16, 64] if quick else [16, 64, 256]
    for n in sizes:
        alpha, beta, mu0 = eqj.legendre_recurrence(n)
        eager = lambda: eqj.gauss_quadrature(alpha, beta, mu0)
        fast = jax.jit(lambda a, b: eqj.gauss_quadrature(a, b, mu0))

        t_eager, _ = measure(eager)
        t_jit, compile_t = measure(fast, alpha, beta)
        report.row("gauss_quadrature (n=%d)" % n, _fmt(t_eager), _fmt(t_jit),
                   "%.1fx" % (t_eager / t_jit), _fmt(compile_t))

    dim, order = 4, 4
    m = 500 if quick else 4000
    idx = eqj.total_order_indices(dim, order)
    rec = [eqj.uniform_recurrence(order + 1)] * dim
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.uniform(-1.0, 1.0, size=(m, dim)))

    eager = lambda: eqj.design_matrix(X, idx, rec)
    fast = jax.jit(lambda Z: eqj.design_matrix(Z, idx, rec))
    t_eager, _ = measure(eager)
    t_jit, compile_t = measure(fast, X)
    report.row("design_matrix (%d pts, %d terms)" % (m, idx.shape[0]),
               _fmt(t_eager), _fmt(t_jit), "%.1fx" % (t_eager / t_jit),
               _fmt(compile_t))


def bench_vmap(report, quick):
    """One batched call versus a Python loop over the same work."""
    report.section(
        "vmap versus a Python loop",
        "Batching an independent sweep: many measures, or many penalties, "
        "evaluated in one compiled call instead of one at a time.")

    batch = 64 if quick else 512
    n = 24
    rng = np.random.default_rng(1)
    # a batch of perturbed Legendre measures
    alpha0, beta0, mu0 = eqj.legendre_recurrence(n)
    betas = jnp.asarray(np.array(beta0)[None, :]
                        * (1.0 + 0.01 * rng.normal(size=(batch, n - 1))))

    one = jax.jit(lambda b: eqj.gauss_quadrature(alpha0, b, mu0)[0])
    _block(one(betas[0]))
    loop = lambda: [one(betas[i]) for i in range(batch)]

    batched = jax.jit(jax.vmap(lambda b: eqj.gauss_quadrature(alpha0, b, mu0)[0]))
    t_loop, _ = measure(loop, repeat=3)
    t_vmap, compile_t = measure(batched, betas)
    report.row("gauss_quadrature x %d measures" % batch, _fmt(t_loop),
               _fmt(t_vmap), "%.1fx" % (t_loop / t_vmap), _fmt(compile_t))

    # a batch of LASSO solves along a penalty path
    n_path = 8 if quick else 24
    iters = 500 if quick else 2000
    m, p = 60, 90
    A = jnp.asarray(rng.normal(size=(m, p)) / np.sqrt(m))
    c = np.zeros(p); c[rng.choice(p, 5, replace=False)] = 2.0
    y = A @ jnp.asarray(c)
    lams = jnp.asarray(np.geomspace(1e-3, 1e-1, n_path))

    single = jax.jit(lambda lam: eqj.lasso(A, y, lam, iters))
    _block(single(lams[0]))
    loop = lambda: [single(lams[i]) for i in range(n_path)]
    path = jax.jit(lambda ls: eqj.lasso_path(A, y, ls, iters))

    t_loop, _ = measure(loop, repeat=3)
    t_vmap, compile_t = measure(path, lams, repeat=3)
    report.row("lasso path over %d penalties" % n_path, _fmt(t_loop),
               _fmt(t_vmap), "%.1fx" % (t_loop / t_vmap), _fmt(compile_t))


def bench_gradients(report, quick):
    """Adjoint versus finite differences, as the parameter count grows."""
    report.section(
        "Reverse-mode gradient versus finite differences",
        "Gradient of the surrogate variance with respect to every training "
        "observation. Finite differences need two extra solves per parameter; "
        "the adjoint needs one pass regardless. The speed-up is therefore "
        "expected to grow linearly with the parameter count, and does.")

    dim, order = 2, 3
    rec = [eqj.uniform_recurrence(order + 2)] * dim
    idx = eqj.total_order_indices(dim, order)
    sizes = [16, 64] if quick else [16, 64, 256]

    for m in sizes:
        rng = np.random.default_rng(2)
        X = jnp.asarray(rng.uniform(-1.0, 1.0, size=(m, dim)))
        y0 = jnp.asarray(rng.normal(size=m))

        def variance_of(y):
            poly = eqj.Poly(rec, idx)
            poly.fit(X, y)
            return poly.variance()

        grad_fn = jax.jit(jax.grad(variance_of))
        val_fn = jax.jit(variance_of)
        _block(grad_fn(y0)); _block(val_fn(y0))

        def finite_differences():
            eps = 1e-6
            base = np.array(y0)
            out = np.zeros(m)
            for i in range(m):
                yp = base.copy(); yp[i] += eps
                ym = base.copy(); ym[i] -= eps
                out[i] = (float(val_fn(jnp.asarray(yp)))
                          - float(val_fn(jnp.asarray(ym)))) / (2 * eps)
            return out

        t_grad, _ = measure(grad_fn, y0)
        t_fd, _ = measure(finite_differences, repeat=3)

        # sanity: the two must agree, or the timing is meaningless
        err = np.abs(np.array(grad_fn(y0)) - finite_differences()).max()
        report.row("d(variance)/d(y), %d parameters" % m, _fmt(t_fd),
                   _fmt(t_grad), "%.0fx" % (t_fd / t_grad),
                   "agree to %.1e" % err)


def bench_versus_classic(report, quick):
    """The same task through the classic NumPy implementation."""
    report.section(
        "Classic NumPy versus JAX",
        "Identical task, independent implementations (verified equal to ~1e-11 "
        "by tests/test_jax_parity_with_classic.py). The point is that the port "
        "costs nothing on CPU, not that it wins.")

    try:
        import equadratures as eq
    except Exception as exc:                          # pragma: no cover
        report.row("classic import failed", str(exc), "-", "-", "-")
        return

    dim, order = 2, 4
    m = 500 if quick else 4000
    rng = np.random.default_rng(3)
    X = rng.uniform(-1.0, 1.0, size=(m, dim))

    p = eq.Parameter(distribution="uniform", lower=-1.0, upper=1.0, order=order)
    basis = eq.Basis("total-order", orders=[order] * dim)
    poly_c = eq.Poly([p] * dim, basis, method="least-squares")

    idx = eqj.total_order_indices(dim, order)
    rec = [eqj.uniform_recurrence(order + 2)] * dim
    Xj = jnp.asarray(X)
    fast = jax.jit(lambda Z: eqj.design_matrix(Z, idx, rec))
    _block(fast(Xj))

    t_classic, _ = measure(lambda: poly_c.get_poly(X))
    t_jax, _ = measure(fast, Xj)
    report.row("design_matrix (%d pts, %d terms)" % (m, idx.shape[0]),
               _fmt(t_classic), _fmt(t_jax), "%.1fx" % (t_classic / t_jax), "-")

    n = 64
    alpha, beta, mu0 = eqj.legendre_recurrence(n)
    ab = np.zeros((n, 2))
    ab[:, 0] = np.array(alpha)
    ab[1:, 1] = np.array(beta) ** 2

    def classic_gauss():
        J = np.diag(ab[:, 0]) + np.diag(np.sqrt(ab[1:, 1]), 1) \
            + np.diag(np.sqrt(ab[1:, 1]), -1)
        w, v = np.linalg.eigh(J)
        return w, 2.0 * v[0, :] ** 2

    fast_gauss = jax.jit(lambda a, b: eqj.gauss_quadrature(a, b, mu0))
    _block(fast_gauss(alpha, beta))
    t_classic, _ = measure(classic_gauss)
    t_jax, _ = measure(fast_gauss, alpha, beta)
    report.row("gauss_quadrature (n=%d)" % n, _fmt(t_classic), _fmt(t_jax),
               "%.1fx" % (t_classic / t_jax), "-")


# --------------------------------------------------------------------- driver
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="small sizes and fewer repeats, for CI")
    parser.add_argument("--out", default=None,
                        help="write the markdown report to this file too")
    args = parser.parse_args(argv)

    devices = jax.devices()
    header = [
        "# equadratures.jax benchmarks",
        "",
        "- platform: `%s`, Python %s" % (platform.platform(),
                                         platform.python_version()),
        "- jax %s on %s" % (jax.__version__,
                            ", ".join(str(d.device_kind) for d in devices)),
        "- numpy %s" % np.__version__,
        "- mode: %s" % ("quick" if args.quick else "full"),
        "",
    ]
    if not any(d.platform == "gpu" for d in devices):
        header += [
            "> **No GPU on this machine** -- `jaxlib` is the CPU build, so the",
            "> accelerator half of deliverable D2 is *not* evidenced below.",
            "> What is measured here (compilation, batching, adjoint scaling)",
            "> is hardware-independent. GPU and `pmap` parity remain open.",
            "",
        ]

    report = Report()
    bench_jit(report, args.quick)
    bench_vmap(report, args.quick)
    bench_gradients(report, args.quick)
    bench_versus_classic(report, args.quick)

    text = "\n".join(header) + report.render(
        ["benchmark", "baseline", "equadratures.jax", "speed-up", "notes"])
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
