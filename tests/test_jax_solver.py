"""Validation for the differentiable sparse / regularised solvers.

Acceptance gate, in two halves.

*Correctness of the solve*: the LASSO recovers a known sparse coefficient vector
from an under-determined system (the compressed-sensing claim), the elastic net
reduces to ridge when the ``l1`` penalty vanishes, and the converged point
satisfies the sub-gradient optimality conditions.

*Correctness of the gradient*: implicit differentiation is checked against
finite differences w.r.t. the data, the design matrix and both penalties, and
against the closed-form Jacobian in the ridge limit where one exists. Gradients
are also shown to be independent of the FISTA iteration count -- the property
that distinguishes implicit differentiation from unrolling.
"""
import unittest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import equadratures.jax as eqj
    from equadratures.jax.solver import elastic_net, lasso, ridge, soft_threshold
    _HAS_JAX = True
except Exception:                                    # pragma: no cover
    _HAS_JAX = False


def _problem(m=24, n=40, k=5, seed=0):
    """Under-determined system with a known ``k``-sparse solution."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n)) / np.sqrt(m)
    c_true = np.zeros(n)
    support = rng.choice(n, size=k, replace=False)
    c_true[support] = np.sign(rng.normal(size=k)) * (1.0 + rng.uniform(size=k))
    y = A @ c_true
    return jnp.asarray(A), jnp.asarray(y), c_true, support


@unittest.skipUnless(_HAS_JAX, "jax not installed (pip install equadratures[jax])")
class TestJaxSolver(unittest.TestCase):

    def test_soft_threshold(self):
        v = jnp.asarray([-3.0, -0.5, 0.0, 0.5, 3.0])
        np.testing.assert_allclose(np.array(soft_threshold(v, 1.0)),
                                   [-2.0, 0.0, 0.0, 0.0, 2.0], atol=1e-14)

    def test_lasso_recovers_sparse_signal(self):
        # Compressed sensing in a regime that satisfies the irrepresentable
        # condition (60 samples, 80 unknowns, 5 non-zeros): the active set is
        # recovered exactly.
        A, y, c_true, support = _problem(m=60, n=80, k=5, seed=0)
        c = np.array(lasso(A, y, 3e-3, max_iter=10000))
        self.assertEqual(set(np.flatnonzero(c)), set(support))
        np.testing.assert_allclose(c, c_true, atol=2e-2)

    def test_lasso_always_covers_the_true_support(self):
        # Harder regime (24 x 40, 5 non-zeros): exact support recovery is not
        # guaranteed -- LASSO needs mutual incoherence for that -- but no true
        # non-zero is ever dropped, and the answer stays far sparser than n.
        A, y, c_true, support = _problem()
        c = np.array(lasso(A, y, 3e-3, max_iter=10000))
        recovered = set(np.flatnonzero(c))
        self.assertTrue(set(support) <= recovered)
        self.assertLess(len(recovered), A.shape[0])

    def test_lasso_bias_vanishes_with_penalty(self):
        # The l1 penalty shrinks surviving coefficients by O(lam); the error
        # must therefore fall roughly linearly as lam -> 0.
        A, y, c_true, _ = _problem(m=60, n=80, k=5, seed=0)
        errs = [np.abs(np.array(lasso(A, y, lam, max_iter=10000)) - c_true).max()
                for lam in (2e-2, 2e-3)]
        self.assertLess(errs[1], 0.2 * errs[0])

    def test_debiasing_removes_shrinkage_bias(self):
        # Re-fitting on the selected support recovers the coefficients to
        # near machine precision, far better than the penalised solve.
        A, y, c_true, support = _problem(m=60, n=80, k=5, seed=0)
        lam = 2e-2
        c_l1 = np.array(lasso(A, y, lam, max_iter=10000))
        c_db = np.array(eqj.lasso_debiased(A, y, lam, max_iter=10000))
        self.assertEqual(set(np.flatnonzero(c_db)), set(support))
        self.assertLess(np.abs(c_db - c_true).max(), 1e-10)
        self.assertLess(np.abs(c_db - c_true).max(),
                        1e-3 * np.abs(c_l1 - c_true).max())

    def test_debiased_gradient_wrt_data(self):
        # Support is piecewise constant, so only the re-fit carries gradient.
        A, y, _, _ = _problem(m=60, n=80, k=5, seed=0)

        def scalar(yv):
            return jnp.sum(eqj.lasso_debiased(A, yv, 2e-2, max_iter=8000) ** 2)

        g = np.array(jax.grad(scalar)(y))
        eps = 1e-6
        yn = np.array(y)
        rng = np.random.default_rng(0)
        for _ in range(6):
            i = int(rng.integers(len(yn)))
            yp = yn.copy(); yp[i] += eps
            ym = yn.copy(); ym[i] -= eps
            fd = (float(scalar(jnp.asarray(yp)))
                  - float(scalar(jnp.asarray(ym)))) / (2 * eps)
            self.assertAlmostEqual(g[i], fd, delta=2e-4 * max(1.0, abs(fd)))

    def test_lasso_is_sparser_than_least_squares(self):
        A, y, _, _ = _problem()
        c_l1 = np.array(lasso(A, y, 1e-2, max_iter=3000))
        c_ls = np.array(jnp.linalg.lstsq(A, y, rcond=None)[0])
        self.assertLess(np.count_nonzero(np.abs(c_l1) > 1e-10),
                        np.count_nonzero(np.abs(c_ls) > 1e-10))

    def test_optimality_conditions(self):
        # Sub-gradient conditions: |A^T(Ac-y)| = lam on the active set,
        # <= lam off it.
        A, y, _, _ = _problem()
        lam = 5e-3
        c = lasso(A, y, lam, max_iter=8000)
        g = np.array(A.T @ (A @ c - y))
        active = np.array(c) != 0.0
        np.testing.assert_allclose(np.abs(g[active]), lam, atol=1e-6)
        self.assertTrue(np.all(np.abs(g[~active]) <= lam + 1e-6))

    def test_elastic_net_reduces_to_ridge(self):
        # With no l1 penalty the elastic net is a plain ridge solve.
        A, y, _, _ = _problem()
        c_en = np.array(elastic_net(A, y, 0.0, 0.1, max_iter=8000))
        c_ridge = np.array(ridge(A, y, 0.1))
        np.testing.assert_allclose(c_en, c_ridge, atol=1e-7)

    def test_gradient_wrt_data_matches_finite_differences(self):
        A, y, _, _ = _problem(m=20, n=30, k=4, seed=1)
        lam = 1e-2

        def scalar(yv):
            return jnp.sum(lasso(A, yv, lam, max_iter=6000) ** 2)

        g = np.array(jax.grad(scalar)(y))
        eps = 1e-6
        yn = np.array(y)
        fd = np.zeros_like(yn)
        for i in range(len(yn)):
            yp = yn.copy(); yp[i] += eps
            ym = yn.copy(); ym[i] -= eps
            fd[i] = (float(scalar(jnp.asarray(yp)))
                     - float(scalar(jnp.asarray(ym)))) / (2 * eps)
        np.testing.assert_allclose(g, fd, rtol=2e-4, atol=1e-7)

    def test_gradient_wrt_penalties_matches_finite_differences(self):
        A, y, _, _ = _problem(m=20, n=30, k=4, seed=2)

        def scalar(l1, l2):
            return jnp.sum(elastic_net(A, y, l1, l2, max_iter=6000) ** 2)

        l1_0, l2_0 = 1e-2, 5e-3
        g1, g2 = jax.grad(scalar, argnums=(0, 1))(l1_0, l2_0)
        eps = 1e-7
        fd1 = (float(scalar(l1_0 + eps, l2_0))
               - float(scalar(l1_0 - eps, l2_0))) / (2 * eps)
        fd2 = (float(scalar(l1_0, l2_0 + eps))
               - float(scalar(l1_0, l2_0 - eps))) / (2 * eps)
        self.assertAlmostEqual(float(g1), fd1, delta=2e-4 * max(1.0, abs(fd1)))
        self.assertAlmostEqual(float(g2), fd2, delta=2e-4 * max(1.0, abs(fd2)))

    def test_gradient_wrt_design_matches_finite_differences(self):
        A, y, _, _ = _problem(m=16, n=24, k=3, seed=3)
        lam = 1e-2

        def scalar(Av):
            return jnp.sum(lasso(Av, y, lam, max_iter=6000) ** 2)

        G = np.array(jax.grad(scalar)(A))
        eps = 1e-6
        An = np.array(A)
        # spot-check a handful of entries: the full m*n sweep is needlessly slow
        rng = np.random.default_rng(0)
        for _ in range(8):
            i = int(rng.integers(An.shape[0]))
            j = int(rng.integers(An.shape[1]))
            Ap = An.copy(); Ap[i, j] += eps
            Am = An.copy(); Am[i, j] -= eps
            fd = (float(scalar(jnp.asarray(Ap)))
                  - float(scalar(jnp.asarray(Am)))) / (2 * eps)
            self.assertAlmostEqual(G[i, j], fd, delta=2e-4 * max(1.0, abs(fd)))

    def test_ridge_limit_gradient_matches_closed_form(self):
        # lam_l1 = 0 : c = (A^T A + lam I)^{-1} A^T y, so dc/dy is known exactly.
        A, y, _, _ = _problem(m=20, n=12, k=3, seed=4)
        lam = 0.3
        n = A.shape[1]
        jac = np.array(jax.jacobian(lambda yv: elastic_net(A, yv, 0.0, lam,
                                                           8000))(y))
        closed = np.linalg.solve(np.array(A.T @ A) + lam * np.eye(n),
                                 np.array(A.T))
        np.testing.assert_allclose(jac, closed, atol=1e-6)

    def test_gradient_independent_of_iteration_count(self):
        # The signature of implicit differentiation: once converged, more
        # iterations change nothing. Unrolling would drift.
        A, y, _, _ = _problem(m=20, n=30, k=4, seed=5)

        def scalar(yv, iters):
            return jnp.sum(lasso(A, yv, 1e-2, max_iter=iters) ** 2)

        g_few = np.array(jax.grad(scalar)(y, 4000))
        g_many = np.array(jax.grad(scalar)(y, 12000))
        np.testing.assert_allclose(g_few, g_many, rtol=1e-9, atol=1e-11)

    def test_jit_and_vmap(self):
        A, y, _, _ = _problem(m=20, n=30, k=4, seed=6)
        fast = jax.jit(lambda yv: lasso(A, yv, 1e-2, max_iter=3000))
        np.testing.assert_allclose(np.array(fast(y)),
                                   np.array(lasso(A, y, 1e-2, max_iter=3000)),
                                   atol=1e-12)
        path = eqj.lasso_path(A, y, jnp.asarray([1e-3, 1e-2, 1e-1]), 3000)
        self.assertEqual(path.shape, (3, A.shape[1]))
        # heavier penalty => sparser solution
        nnz = [int(np.count_nonzero(np.array(row))) for row in path]
        self.assertGreaterEqual(nnz[0], nnz[1])
        self.assertGreaterEqual(nnz[1], nnz[2])

    def test_poly_lasso_fit_recovers_sparse_expansion(self):
        # A polynomial that is sparse in the orthonormal basis, fitted from
        # fewer samples than there are basis terms.
        rec = [eqj.uniform_recurrence(8), eqj.uniform_recurrence(8)]
        idx = eqj.total_order_indices(2, 6)          # 28 basis terms
        rng = np.random.default_rng(7)
        X = jnp.asarray(rng.uniform(-1.0, 1.0, size=(20, 2)))   # 20 samples
        f = lambda Z: 1.0 + 2.0 * Z[:, 0] + 3.0 * Z[:, 0] * Z[:, 1]

        poly = eqj.Poly(rec, idx)
        c = np.array(poly.fit_lasso(X, f(X), 1e-3, max_iter=8000))
        # far sparser than the basis, and sparse enough to be determined by the
        # 20 available samples -- which plain least squares could not be
        self.assertLess(np.count_nonzero(c), X.shape[0])
        Xt = jnp.asarray(rng.uniform(-1.0, 1.0, size=(15, 2)))
        np.testing.assert_allclose(np.array(poly.predict(Xt)),
                                   np.array(f(Xt)), atol=1e-2)

    def test_poly_elastic_net_penalties_are_learnable(self):
        # The point of differentiable penalties: gradient of a validation loss
        # w.r.t. (l1, l2) exists and is finite, so they can be trained.
        rec = [eqj.uniform_recurrence(6), eqj.uniform_recurrence(6)]
        idx = eqj.total_order_indices(2, 4)
        rng = np.random.default_rng(8)
        X = jnp.asarray(rng.uniform(-1.0, 1.0, size=(30, 2)))
        Xv = jnp.asarray(rng.uniform(-1.0, 1.0, size=(20, 2)))
        f = lambda Z: 1.0 + 2.0 * Z[:, 0] + 3.0 * Z[:, 0] * Z[:, 1]

        def val_loss(l1, l2):
            p = eqj.Poly(rec, idx)
            p.fit_elastic_net(X, f(X), l1, l2, max_iter=4000)
            return jnp.mean((p.predict(Xv) - f(Xv)) ** 2)

        g1, g2 = jax.grad(val_loss, argnums=(0, 1))(1e-3, 1e-3)
        self.assertTrue(np.isfinite(float(g1)))
        self.assertTrue(np.isfinite(float(g2)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
