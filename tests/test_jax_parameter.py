"""Validation for equadratures.jax Parameter (differentiable distribution map)."""
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
class TestJaxParameter(unittest.TestCase):

    def test_uniform_map_roundtrip(self):
        p = eqj.Parameter("uniform", lower=2.0, upper=6.0)
        xs = jnp.linspace(-1.0, 1.0, 5)
        back = p.to_standard(p.to_physical(xs))
        np.testing.assert_allclose(np.array(back), np.array(xs), atol=1e-12)

    def test_expectation_differentiable_wrt_bound(self):
        # E[X^2] for X ~ Uniform[a, b] = (a^2 + a b + b^2)/3 ; d/db = (a + 2b)/3.
        # Differentiating an *expectation* w.r.t. a distribution parameter is the
        # full auto-differentiable thesis (not just w.r.t. data).
        n, a = 5, -1.0
        alpha, beta, mu0 = eqj.uniform_recurrence(n)
        std_nodes, W = eqj.gauss_quadrature(alpha, beta, mu0)   # W sums to 1

        def E_x2(b):
            p = eqj.Parameter("uniform", lower=a, upper=b)
            phys = p.to_physical(std_nodes)
            return jnp.sum(W * phys ** 2)

        b0 = 6.0
        self.assertAlmostEqual(float(E_x2(b0)), (a * a + a * b0 + b0 * b0) / 3.0, places=8)
        self.assertAlmostEqual(float(jax.grad(E_x2)(b0)), (a + 2.0 * b0) / 3.0, places=6)

    def test_gaussian_recurrence_probability_normalised(self):
        p = eqj.Parameter("gaussian", mean=1.0, std=2.0)
        _, _, mu0 = p.recurrence(4)
        self.assertAlmostEqual(float(mu0), 1.0, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
