"""Validation for measures built from a density -- differentiating through the
*distribution itself*.

This closes the last link in the differentiable chain. Everything else in the
namespace differentiates with respect to data, inputs or basis coefficients;
here the derivative is taken with respect to a **distribution parameter** -- a
Beta shape, a truncation bound -- so questions like "how sensitive is my output
variance to how I modelled the input?" become one ``jax.grad`` call.

Acceptance gate:

* the constructed measure reproduces analytic moments (Beta mean/variance appear
  directly as the first recurrence coefficients);
* it reproduces the *analytic* recurrences when handed the corresponding
  densities (uniform -> Legendre, wide Gaussian -> Hermite);
* gradients with respect to a shape parameter match a **closed-form**
  derivative, not merely finite differences;
* the whole ``Poly`` pipeline differentiates through to a UQ output.
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


def _beta_second_moment(a, b):
    """Exact E[X^2] for X ~ Beta(a, b)."""
    return a * (a + 1.0) / ((a + b) * (a + b + 1.0))


def _beta_second_moment_grad(a, b):
    """Exact d/da E[X^2] for X ~ Beta(a, b)."""
    s = a + b
    N, dN = a * (a + 1.0), 2.0 * a + 1.0
    D, dD = s * s + s, 2.0 * s + 1.0
    return (dN * D - N * dD) / D ** 2


@unittest.skipUnless(_HAS_JAX, "jax not installed (pip install equadratures[jax])")
class TestJaxDensityParameter(unittest.TestCase):

    def test_beta_moments_appear_in_the_recurrence(self):
        # For a probability measure the first recurrence coefficients are the
        # mean and the variance: alpha_0 = E[X], beta_0^2 = Var[X].
        a, b = 2.0, 3.0
        alpha, beta, mu0 = eqj.density_recurrence(
            eqj.beta_density(a, b), 0.0, 1.0, n=6, n_grid=300)
        self.assertAlmostEqual(float(mu0), 1.0, places=12)
        self.assertAlmostEqual(float(alpha[0]), a / (a + b), places=12)
        var = a * b / ((a + b) ** 2 * (a + b + 1.0))
        self.assertAlmostEqual(float(beta[0] ** 2), var, places=12)

    def test_uniform_density_recovers_legendre(self):
        # Handed a flat density, the numerical construction must reproduce the
        # analytic uniform recurrence exactly.
        n = 8
        alpha, beta, mu0 = eqj.density_recurrence(
            lambda x: jnp.ones_like(x), -1.0, 1.0, n=n, n_grid=200)
        a_ref, b_ref, mu_ref = eqj.uniform_recurrence(n)
        np.testing.assert_allclose(np.array(alpha), np.array(a_ref), atol=1e-11)
        np.testing.assert_allclose(np.array(beta), np.array(b_ref), atol=1e-11)
        self.assertAlmostEqual(float(mu0), float(mu_ref), places=12)

    def test_wide_gaussian_density_recovers_hermite(self):
        # Truncating a standard normal at +-9 sigma loses nothing measurable, so
        # the numerical recurrence must match the analytic Hermite one.
        n = 6
        alpha, beta, mu0 = eqj.density_recurrence(
            eqj.truncated_gaussian_density(0.0, 1.0), -9.0, 9.0, n=n, n_grid=400)
        a_ref, b_ref, _ = eqj.hermite_recurrence(n)
        np.testing.assert_allclose(np.array(alpha), np.array(a_ref), atol=1e-9)
        np.testing.assert_allclose(np.array(beta), np.array(b_ref), atol=1e-9)

    def test_quadrature_integrates_beta_moments(self):
        a, b = 2.5, 3.0
        alpha, beta, mu0 = eqj.density_recurrence(
            eqj.beta_density(a, b), 0.0, 1.0, n=8, n_grid=400)
        nodes, weights = eqj.gauss_quadrature(alpha, beta, mu0)
        self.assertAlmostEqual(float(jnp.sum(weights)), 1.0, places=12)
        self.assertAlmostEqual(float(jnp.sum(weights * nodes ** 2)),
                               _beta_second_moment(a, b), places=10)

    # -------------------------------------------------- the headline gradient
    def test_gradient_wrt_shape_parameter_matches_closed_form(self):
        def second_moment(a):
            alpha, beta, mu0 = eqj.density_recurrence(
                eqj.beta_density(a, 3.0), 0.0, 1.0, n=8, n_grid=300)
            nodes, weights = eqj.gauss_quadrature(alpha, beta, mu0)
            return jnp.sum(weights * nodes ** 2)

        for a in (2.0, 2.5, 3.0):
            g = float(jax.grad(second_moment)(a))
            self.assertAlmostEqual(g, _beta_second_moment_grad(a, 3.0),
                                   places=8,
                                   msg="d/da E[X^2] wrong at a=%s" % a)

    def test_gradient_wrt_truncation_bound(self):
        # Differentiating with respect to the *support* of the measure.
        def variance(upper):
            alpha, beta, mu0 = eqj.density_recurrence(
                eqj.truncated_gaussian_density(0.0, 1.0), -3.0, upper,
                n=6, n_grid=300)
            nodes, weights = eqj.gauss_quadrature(alpha, beta, mu0)
            m = jnp.sum(weights * nodes)
            return jnp.sum(weights * (nodes - m) ** 2)

        u0 = 2.0
        g = float(jax.grad(variance)(u0))
        eps = 1e-5
        fd = (float(variance(u0 + eps)) - float(variance(u0 - eps))) / (2 * eps)
        self.assertAlmostEqual(g, fd, delta=1e-5 * max(1.0, abs(fd)))
        self.assertNotAlmostEqual(g, 0.0, places=6)   # genuinely depends on it

    def test_unbounded_density_where_classic_fails(self):
        # Beta(0.5, 0.5) diverges at both endpoints. Classic's equispaced pdf
        # grid evaluates the endpoints and yields inf; Gauss-Legendre nodes are
        # strictly interior, so the measure is still usable. Convergence is
        # algebraic rather than spectral, hence the looser tolerance on the
        # variance -- that is the honest accuracy, not a fudge.
        alpha, beta, mu0 = eqj.density_recurrence(
            eqj.beta_density(0.5, 0.5), 0.0, 1.0, n=6, n_grid=400)
        self.assertTrue(np.all(np.isfinite(np.array(alpha))))
        self.assertTrue(np.all(np.isfinite(np.array(beta))))
        nodes, weights = eqj.gauss_quadrature(alpha, beta, mu0)
        self.assertAlmostEqual(float(jnp.sum(weights * nodes)), 0.5, places=10)
        var = float(jnp.sum(weights * (nodes - 0.5) ** 2))
        self.assertAlmostEqual(var, 0.125, delta=1e-3)

    # ------------------------------------------------- end-to-end through Poly
    def test_density_parameter_drives_poly(self):
        p = eqj.DensityParameter(eqj.beta_density(2.0, 3.0), 0.0, 1.0)
        rec = [p.recurrence(6)]
        X, W = eqj.tensor_quadrature(rec)
        self.assertAlmostEqual(float(jnp.sum(W)), 1.0, places=12)

        # E[X] and E[X^2] through the Poly layer agree with Beta analytics
        poly = eqj.Poly(rec, eqj.total_order_indices(1, 3))
        poly.fit_projection(X, X[:, 0] ** 2, W)
        self.assertAlmostEqual(float(poly.mean()),
                               _beta_second_moment(2.0, 3.0), places=10)

    def test_uq_output_differentiable_wrt_distribution_parameter(self):
        # The end-to-end claim: d(surrogate variance)/d(input distribution
        # shape). Nothing in the classic namespace can produce this number.
        f = lambda Z: jnp.exp(Z[:, 0]) + Z[:, 0] ** 3

        def variance_given(a):
            p = eqj.DensityParameter(eqj.beta_density(a, 3.0), 0.0, 1.0,
                                     n_grid=250)
            rec = [p.recurrence(8)]
            X, W = eqj.tensor_quadrature(rec)
            poly = eqj.Poly(rec, eqj.total_order_indices(1, 5))
            poly.fit_projection(X, f(X), W)
            return poly.variance()

        a0 = 2.0
        g = float(jax.grad(variance_given)(a0))
        eps = 1e-5
        fd = (float(variance_given(a0 + eps))
              - float(variance_given(a0 - eps))) / (2 * eps)
        self.assertTrue(np.isfinite(g))
        self.assertAlmostEqual(g, fd, delta=1e-4 * max(1.0, abs(fd)))
        self.assertNotAlmostEqual(g, 0.0, places=6)

    def test_sobol_indices_differentiable_wrt_distribution_parameter(self):
        # Same idea, on a genuinely multivariate UQ output.
        f = lambda Z: Z[:, 0] + 2.0 * Z[:, 1] + 3.0 * Z[:, 0] * Z[:, 1]

        def first_sobol(a):
            p = eqj.DensityParameter(eqj.beta_density(a, 3.0), 0.0, 1.0,
                                     n_grid=200)
            rec = [p.recurrence(5), eqj.uniform_recurrence(5)]
            X, W = eqj.tensor_quadrature(rec)
            poly = eqj.Poly(rec, eqj.total_order_indices(2, 3))
            poly.fit_projection(X, f(X), W)
            return poly.sobol_indices()[0]

        a0 = 2.0
        g = float(jax.grad(first_sobol)(a0))
        eps = 1e-5
        fd = (float(first_sobol(a0 + eps))
              - float(first_sobol(a0 - eps))) / (2 * eps)
        self.assertAlmostEqual(g, fd, delta=1e-4 * max(1.0, abs(fd)))

    def test_grid_is_cached(self):
        # The Legendre discretisation is a constant; it must not be rebuilt on
        # every gradient step.
        from equadratures.jax.parameter import _legendre_grid
        _legendre_grid.cache_clear()
        _legendre_grid(64)
        _legendre_grid(64)
        info = _legendre_grid.cache_info()
        self.assertEqual(info.misses, 1)
        self.assertEqual(info.hits, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
