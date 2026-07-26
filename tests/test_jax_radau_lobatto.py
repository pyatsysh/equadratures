"""Validation for the Radau and Lobatto rules with prescribed nodes.

Acceptance gate, same shape as the Gauss rules: an exactness (sum-rule) check
at the *correct reduced degree*, textbook node/weight values, the prescribed
nodes actually appearing in the rule, and autodiff against finite differences.

The reduced degrees are the whole point of these rules, so the tests assert them
sharply -- exact at the theoretical degree and *not* exact one degree above.
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


def _legendre_moment(p):
    """Integral of x^p over [-1, 1] with unit weight."""
    return 0.0 if p % 2 else 2.0 / (p + 1)


@unittest.skipUnless(_HAS_JAX, "jax not installed (pip install equadratures[jax])")
class TestJaxRadauLobatto(unittest.TestCase):

    def _max_exact_degree(self, nodes, weights, tol=1e-10):
        """Highest ``d`` such that every degree ``<= d`` integrates exactly."""
        d = -1
        while True:
            approx = float(jnp.sum(weights * nodes ** (d + 1)))
            if abs(approx - _legendre_moment(d + 1)) > tol:
                return d
            d += 1

    def test_radau_nodes_and_weights(self):
        # 3-point Gauss-Radau on [-1, 1] fixed at -1 (textbook values):
        # nodes -1, (1 -+ sqrt 6)/5 ; weights 2/9, (16 +- sqrt 6)/18.
        alpha, beta, mu0 = eqj.legendre_recurrence(3)
        nodes, weights = eqj.radau_quadrature(alpha, beta, mu0, -1.0)
        s6 = np.sqrt(6.0)
        np.testing.assert_allclose(np.array(nodes),
                                   [-1.0, (1 - s6) / 5, (1 + s6) / 5], atol=1e-12)
        np.testing.assert_allclose(np.array(weights),
                                   [2 / 9, (16 + s6) / 18, (16 - s6) / 18],
                                   atol=1e-12)

    def test_lobatto_nodes_and_weights(self):
        # 4-point Gauss-Lobatto: nodes +-1, +-1/sqrt(5); weights 1/6, 5/6.
        alpha, beta, mu0 = eqj.legendre_recurrence(4)
        nodes, weights = eqj.lobatto_quadrature(alpha, beta, mu0, -1.0, 1.0)
        r5 = 1.0 / np.sqrt(5.0)
        np.testing.assert_allclose(np.array(nodes), [-1.0, -r5, r5, 1.0],
                                   atol=1e-12)
        np.testing.assert_allclose(np.array(weights),
                                   [1 / 6, 5 / 6, 5 / 6, 1 / 6], atol=1e-12)

        # 5-point: nodes 0, +-sqrt(3/7), +-1 ; weights 32/45, 49/90, 1/10.
        alpha, beta, mu0 = eqj.legendre_recurrence(5)
        nodes, weights = eqj.lobatto_quadrature(alpha, beta, mu0, -1.0, 1.0)
        r = np.sqrt(3.0 / 7.0)
        np.testing.assert_allclose(np.array(nodes), [-1.0, -r, 0.0, r, 1.0],
                                   atol=1e-12)
        np.testing.assert_allclose(np.array(weights),
                                   [1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10],
                                   atol=1e-12)

    def test_prescribed_nodes_are_included(self):
        for n in (3, 5, 8):
            alpha, beta, mu0 = eqj.legendre_recurrence(n)
            nodes, _ = eqj.radau_quadrature(alpha, beta, mu0, -1.0)
            self.assertAlmostEqual(float(nodes[0]), -1.0, places=12)

            nodes, _ = eqj.radau_quadrature(alpha, beta, mu0, 1.0)
            self.assertAlmostEqual(float(nodes[-1]), 1.0, places=12)

            nodes, _ = eqj.lobatto_quadrature(alpha, beta, mu0, -1.0, 1.0)
            self.assertAlmostEqual(float(nodes[0]), -1.0, places=12)
            self.assertAlmostEqual(float(nodes[-1]), 1.0, places=12)

    def test_exactness_degrees(self):
        # The defining property: Gauss 2n-1, Radau 2n-2, Lobatto 2n-3 -- and
        # each one *fails* at the next degree up, so the rules are not
        # accidentally better than claimed.
        for n in (4, 5, 6):
            alpha, beta, mu0 = eqj.legendre_recurrence(n)

            nodes, weights = eqj.gauss_quadrature(alpha, beta, mu0)
            self.assertEqual(self._max_exact_degree(nodes, weights), 2 * n - 1)

            nodes, weights = eqj.radau_quadrature(alpha, beta, mu0, -1.0)
            self.assertEqual(self._max_exact_degree(nodes, weights), 2 * n - 2)

            nodes, weights = eqj.lobatto_quadrature(alpha, beta, mu0, -1.0, 1.0)
            self.assertEqual(self._max_exact_degree(nodes, weights), 2 * n - 3)

    def test_weights_are_positive_and_sum_to_mass(self):
        for n in (3, 6, 10):
            alpha, beta, mu0 = eqj.uniform_recurrence(n)     # probability measure
            for nodes, weights in (
                eqj.radau_quadrature(alpha, beta, mu0, -1.0),
                eqj.lobatto_quadrature(alpha, beta, mu0, -1.0, 1.0),
            ):
                self.assertAlmostEqual(float(jnp.sum(weights)), 1.0, places=12)
                self.assertTrue(bool(jnp.all(weights > 0)))

    def test_radau_on_a_non_standard_prescribed_node(self):
        # The construction does not require the node to be an endpoint.
        alpha, beta, mu0 = eqj.legendre_recurrence(5)
        nodes, weights = eqj.radau_quadrature(alpha, beta, mu0, -0.5)
        self.assertTrue(np.any(np.isclose(np.array(nodes), -0.5, atol=1e-12)))
        self.assertAlmostEqual(float(jnp.sum(weights)), 2.0, places=12)

    def test_differentiable_wrt_prescribed_node(self):
        # d/d(end) of a Radau-integrated function, autodiff vs finite difference.
        alpha, beta, mu0 = eqj.legendre_recurrence(5)

        def integral(end):
            nodes, weights = eqj.radau_quadrature(alpha, beta, mu0, end)
            return jnp.sum(weights * jnp.exp(nodes))

        e0 = -0.8
        g = float(jax.grad(integral)(e0))
        eps = 1e-6
        fd = (float(integral(e0 + eps)) - float(integral(e0 - eps))) / (2 * eps)
        self.assertAlmostEqual(g, fd, delta=1e-5)

    def test_differentiable_wrt_recurrence(self):
        # Lobatto nodes/weights differentiate w.r.t. the recurrence coefficients,
        # which is what makes prescribed-node rules usable on learnable measures.
        n = 5
        alpha0, beta0, mu0 = eqj.legendre_recurrence(n)

        def integral(beta):
            nodes, weights = eqj.lobatto_quadrature(alpha0, beta, mu0, -1.0, 1.0)
            return jnp.sum(weights * nodes ** 2)

        g = np.array(jax.grad(integral)(beta0))
        eps = 1e-7
        b = np.array(beta0)
        fd = np.zeros_like(b)
        for i in range(len(b)):
            bp = b.copy(); bp[i] += eps
            bm = b.copy(); bm[i] -= eps
            fd[i] = (float(integral(jnp.asarray(bp)))
                     - float(integral(jnp.asarray(bm)))) / (2 * eps)
        np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-8)

    def test_jit(self):
        alpha, beta, mu0 = eqj.legendre_recurrence(6)
        fast = jax.jit(lambda e: eqj.radau_quadrature(alpha, beta, mu0, e)[0])
        np.testing.assert_allclose(
            np.array(fast(-1.0)),
            np.array(eqj.radau_quadrature(alpha, beta, mu0, -1.0)[0]),
            atol=1e-13)

    def test_lobatto_on_gaussian_measure(self):
        # Prescribed nodes on an unbounded measure: truncating a standard normal
        # at +-3 still integrates low-order moments well and reproduces mass.
        n = 8
        alpha, beta, mu0 = eqj.hermite_recurrence(n)
        nodes, weights = eqj.lobatto_quadrature(alpha, beta, mu0, -3.0, 3.0)
        self.assertAlmostEqual(float(jnp.sum(weights)), 1.0, places=12)
        self.assertAlmostEqual(float(jnp.sum(weights * nodes)), 0.0, places=10)
        self.assertAlmostEqual(float(jnp.sum(weights * nodes ** 2)), 1.0, places=8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
