"""Validation for equadratures.jax differentiable Gauss quadrature.

Acceptance gate for a JAX port (see vault roadmap): every primitive must pass a
*sum-rule / exactness* check and an *autodiff-vs-finite-difference* check.
Skipped automatically when JAX is not installed.
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


def _normal_moment(p):
    """E[X^p] for X ~ N(0, 1): 0 if p odd, (p-1)!! if p even."""
    if p % 2 == 1:
        return 0.0
    m = 1.0
    for k in range(1, p, 2):
        m *= k
    return m


@unittest.skipUnless(_HAS_JAX, "jax not installed (pip install equadratures[jax])")
class TestJaxQuadrature(unittest.TestCase):

    def test_legendre_sum_rule(self):
        # n-point Gauss-Legendre integrates degree <= 2n-1 exactly on [-1, 1].
        n = 6
        alpha, beta, mu0 = eqj.legendre_recurrence(n)
        nodes, weights = eqj.gauss_quadrature(alpha, beta, mu0)
        self.assertAlmostEqual(float(jnp.sum(weights)), 2.0, places=12)
        for p in range(0, 2 * n):
            approx = float(jnp.sum(weights * nodes ** p))
            exact = 0.0 if p % 2 == 1 else 2.0 / (p + 1)
            self.assertAlmostEqual(approx, exact, places=10,
                                   msg="Legendre degree %d not exact" % p)

    def test_hermite_sum_rule(self):
        # n-point Gauss-Hermite integrates Gaussian moments to degree 2n-1.
        n = 6
        alpha, beta, mu0 = eqj.hermite_recurrence(n)
        nodes, weights = eqj.gauss_quadrature(alpha, beta, mu0)
        self.assertAlmostEqual(float(jnp.sum(weights)), 1.0, places=12)
        for p in range(0, 2 * n):
            approx = float(jnp.sum(weights * nodes ** p))
            self.assertAlmostEqual(approx, _normal_moment(p), places=8,
                                   msg="Hermite degree %d not exact" % p)

    def test_nodes_differentiable_wrt_beta(self):
        # eigh autodiff: d(node)/d(beta) must match central finite differences.
        n = 5
        alpha, beta, mu0 = eqj.legendre_recurrence(n)

        def largest_node(b):
            nodes, _ = eqj.gauss_quadrature(alpha, b, mu0)
            return nodes[-1]

        g = np.array(jax.grad(largest_node)(beta))
        eps = 1e-6
        b = np.array(beta)
        fd = np.zeros_like(b)
        for i in range(len(b)):
            bp = b.copy(); bp[i] += eps
            bm = b.copy(); bm[i] -= eps
            fd[i] = (float(largest_node(jnp.array(bp)))
                     - float(largest_node(jnp.array(bm)))) / (2 * eps)
        np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)

    def test_integral_grad_wrt_scale(self):
        # Q(s) estimates ∫_{-s}^{s} x^2 dx = 2 s^3 / 3, so dQ/ds = 2 s^2.
        n = 4
        alpha, beta, mu0 = eqj.legendre_recurrence(n)
        std_nodes, std_weights = eqj.gauss_quadrature(alpha, beta, mu0)

        def Q(s):
            return jnp.sum((s * std_weights) * (s * std_nodes) ** 2)

        s0 = 1.7
        self.assertAlmostEqual(float(Q(s0)), 2.0 * s0 ** 3 / 3.0, places=10)
        self.assertAlmostEqual(float(jax.grad(Q)(s0)), 2.0 * s0 ** 2, places=6)

    def test_orthonormality_via_quadrature(self):
        # <p_i, p_j> = sum_q w_q p_i(x_q) p_j(x_q) = delta_ij, evaluated with a
        # Gauss rule exact for products up to degree 2*degree. Ties the
        # quadrature and the polynomial recurrence together (sum-rule checkpoint).
        degree = 5
        nquad = degree + 2
        for recur in (eqj.legendre_recurrence, eqj.hermite_recurrence):
            alpha, beta, mu0 = recur(nquad)
            nodes, weights = eqj.gauss_quadrature(alpha, beta, mu0)
            P = eqj.orthonormal_polynomials(nodes, alpha, beta, mu0, degree)
            gram = (P * weights) @ P.T                    # (degree+1, degree+1)
            np.testing.assert_allclose(np.array(gram), np.eye(degree + 1),
                                       rtol=0, atol=1e-8)

    def test_polynomials_differentiable(self):
        # d/dx p_degree(x) via autodiff matches central finite differences.
        degree = 4
        alpha, beta, mu0 = eqj.legendre_recurrence(degree + 1)

        def p_top(xs):
            P = eqj.orthonormal_polynomials(jnp.array([xs]), alpha, beta, mu0, degree)
            return P[degree, 0]

        x0 = 0.37
        g = float(jax.grad(p_top)(x0))
        eps = 1e-6
        fd = (float(p_top(x0 + eps)) - float(p_top(x0 - eps))) / (2 * eps)
        self.assertAlmostEqual(g, fd, places=6)

    def test_jit_and_vmap(self):
        n = 5
        alpha, beta, mu0 = eqj.legendre_recurrence(n)
        nodes, weights = jax.jit(eqj.gauss_quadrature)(alpha, beta, mu0)
        self.assertEqual(nodes.shape, (n,))
        betas = jnp.stack([beta, 1.5 * beta])
        out = jax.vmap(lambda b: eqj.gauss_quadrature(alpha, b, mu0)[0])(betas)
        self.assertEqual(out.shape, (2, n))


if __name__ == "__main__":
    unittest.main(verbosity=2)
