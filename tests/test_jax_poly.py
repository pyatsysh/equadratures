"""Validation for the equadratures.jax multivariate basis and Poly model.

Acceptance gate: multivariate orthonormality through the tensor quadrature,
exact recovery of a known polynomial (mean/variance correct), regression fit,
and end-to-end differentiability of the surrogate. Skipped if JAX is absent.
"""
import unittest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import equadratures.jax as eqj
    _HAS_JAX = True
except Exception:                                    # pragma: no cover
    _HAS_JAX = False


@unittest.skipUnless(_HAS_JAX, "jax not installed (pip install equadratures[jax])")
class TestJaxPoly(unittest.TestCase):

    def _uniform_2d(self, K):
        return [eqj.uniform_recurrence(K), eqj.uniform_recurrence(K)]

    def test_tensor_quadrature_sum_rule(self):
        # 2-D uniform probability measure: weights sum to 1; E[x1^2 x2^2] = 1/9.
        rec = self._uniform_2d(5)
        X, W = eqj.tensor_quadrature(rec)
        self.assertAlmostEqual(float(jnp.sum(W)), 1.0, places=12)
        val = float(jnp.sum(W * X[:, 0] ** 2 * X[:, 1] ** 2))
        self.assertAlmostEqual(val, 1.0 / 9.0, places=10)

    def test_multivariate_orthonormality(self):
        # A^T diag(W) A = I for a total-order basis on the tensor Gauss grid.
        order = 3
        rec = self._uniform_2d(order + 2)
        idx = eqj.total_order_indices(2, order)
        X, W = eqj.tensor_quadrature(rec)
        A = eqj.design_matrix(X, idx, rec)
        gram = A.T @ (W[:, None] * A)
        np.testing.assert_allclose(np.array(gram), np.eye(idx.shape[0]),
                                   rtol=0, atol=1e-9)

    def test_poly_projection_recovers_polynomial(self):
        # f(x1,x2) = 1 + 2 x1 + 3 x1 x2 on uniform[-1,1]^2.
        # mean = 1, variance = 4/3 + 1 = 7/3, and the surrogate reproduces f.
        rec = self._uniform_2d(4)
        idx = eqj.total_order_indices(2, 2)
        X, W = eqj.tensor_quadrature(rec)
        f = lambda Z: 1.0 + 2.0 * Z[:, 0] + 3.0 * Z[:, 0] * Z[:, 1]
        poly = eqj.Poly(rec, idx)
        poly.fit_projection(X, f(X), W)
        self.assertAlmostEqual(float(poly.mean()), 1.0, places=10)
        self.assertAlmostEqual(float(poly.variance()), 7.0 / 3.0, places=10)
        # reproduces f at arbitrary test points
        Xt = jnp.asarray([[0.3, -0.4], [-0.9, 0.1], [0.5, 0.5]])
        np.testing.assert_allclose(np.array(poly.predict(Xt)), np.array(f(Xt)),
                                   rtol=0, atol=1e-10)

    def test_poly_regression_recovers_polynomial(self):
        # Plain least-squares regression on the grid recovers the same model.
        rec = self._uniform_2d(4)
        idx = eqj.total_order_indices(2, 2)
        X, _ = eqj.tensor_quadrature(rec)
        f = lambda Z: 1.0 + 2.0 * Z[:, 0] + 3.0 * Z[:, 0] * Z[:, 1]
        poly = eqj.Poly(rec, idx)
        poly.fit(X, f(X))
        self.assertAlmostEqual(float(poly.mean()), 1.0, places=8)
        self.assertAlmostEqual(float(poly.variance()), 7.0 / 3.0, places=8)

    def test_sobol_indices(self):
        # f = 1 + 2 x1 + 3 x1 x2 : var = 7/3, main-effect(x1) = 4/3, interaction = 1.
        # S1 = (4/3)/(7/3) = 4/7 ; S2 = 0 ; T1 = 1 ; T2 = (1)/(7/3) = 3/7.
        rec = self._uniform_2d(4)
        idx = eqj.total_order_indices(2, 2)
        X, W = eqj.tensor_quadrature(rec)
        f = lambda Z: 1.0 + 2.0 * Z[:, 0] + 3.0 * Z[:, 0] * Z[:, 1]
        poly = eqj.Poly(rec, idx)
        poly.fit_projection(X, f(X), W)
        S = np.array(poly.sobol_indices())
        T = np.array(poly.total_sobol_indices())
        np.testing.assert_allclose(S, [4.0 / 7.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(T, [1.0, 3.0 / 7.0], atol=1e-9)

    def test_variance_differentiable_wrt_data(self):
        # Differentiable UQ output: d(variance)/d(training y) vs finite differences.
        rec = self._uniform_2d(4)
        idx = eqj.total_order_indices(2, 2)
        X, W = eqj.tensor_quadrature(rec)
        f = lambda Z: 1.0 + 2.0 * Z[:, 0] + 3.0 * Z[:, 0] * Z[:, 1]
        y0 = f(X)

        def variance_of(y):
            poly = eqj.Poly(rec, idx)
            poly.fit_projection(X, y, W)
            return poly.variance()

        g = np.array(jax.grad(variance_of)(y0))
        eps = 1e-6
        y = np.array(y0)
        fd = np.zeros_like(y)
        for i in range(len(y)):
            yp = y.copy(); yp[i] += eps
            ym = y.copy(); ym[i] -= eps
            fd[i] = (float(variance_of(jnp.array(yp)))
                     - float(variance_of(jnp.array(ym)))) / (2 * eps)
        np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)

    def test_poly_ridge_fit(self):
        # Ridge with tiny reg ~ exact fit; and variance differentiable w.r.t. reg.
        rec = self._uniform_2d(4)
        idx = eqj.total_order_indices(2, 2)
        X, _ = eqj.tensor_quadrature(rec)
        f = lambda Z: 1.0 + 2.0 * Z[:, 0] + 3.0 * Z[:, 0] * Z[:, 1]
        poly = eqj.Poly(rec, idx)
        poly.fit_ridge(X, f(X), 1e-12)
        self.assertAlmostEqual(float(poly.mean()), 1.0, places=6)
        self.assertAlmostEqual(float(poly.variance()), 7.0 / 3.0, places=6)

        def var_of(reg):
            p = eqj.Poly(rec, idx)
            p.fit_ridge(X, f(X), reg)
            return p.variance()

        self.assertTrue(np.isfinite(float(jax.grad(var_of)(1e-3))))

    def test_poly_surrogate_differentiable(self):
        # d/dx of the fitted surrogate matches the analytic gradient of f:
        # df/dx1 = 2 + 3 x2 ; df/dx2 = 3 x1.
        rec = self._uniform_2d(4)
        idx = eqj.total_order_indices(2, 2)
        X, W = eqj.tensor_quadrature(rec)
        f = lambda Z: 1.0 + 2.0 * Z[:, 0] + 3.0 * Z[:, 0] * Z[:, 1]
        poly = eqj.Poly(rec, idx)
        poly.fit_projection(X, f(X), W)

        def surrogate(x):                       # x: (2,) -> scalar
            return poly.predict(x[None, :])[0]

        x0 = jnp.asarray([0.3, -0.4])
        g = np.array(jax.grad(surrogate)(x0))
        analytic = np.array([2.0 + 3.0 * (-0.4), 3.0 * 0.3])
        np.testing.assert_allclose(g, analytic, rtol=0, atol=1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
