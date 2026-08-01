"""Invariants swept across configuration space, rather than checked at one point.

Every other test in this namespace pins a specific configuration -- this order,
this dimension, this rule size -- and checks it hard. That leaves a gap: a bug
that only appears at order 9, or only in three dimensions, passes everything.
These tests sweep the same invariants across many configurations instead, which
is a different kind of coverage and cheap to keep.

The invariants are the ones that must hold *by construction* rather than
approximately:

* a Gauss rule has positive weights summing to ``mu0``, distinct finite nodes,
  and is exact to degree ``2n-1``;
* the design matrix is orthonormal under its own quadrature, ``P^T W P = I``;
* a converged ``lasso`` solution cannot be improved by perturbing it;
* projecting and reconstructing a function in the span is the identity, and
  differentiation is represented exactly, at every order;
* a learned ridge subspace is orthonormal and recovers a true ridge direction.

Written after a hand-driven hunt for degenerate inputs found five real defects
elsewhere in the namespace. Notably this sweep found **none** — the failures
were all at pathological inputs (zero variance, exhausted measures, inverted
bounds) rather than at ordinary configurations, which is worth knowing about
where to look next time.

Seeds are fixed, so these are deterministic and safe in CI.
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


@unittest.skipUnless(_HAS_JAX, "jax not installed")
class TestQuadratureInvariants(unittest.TestCase):

    FAMILIES = (("uniform", lambda n: eqj.uniform_recurrence(n)),
                ("legendre", lambda n: eqj.legendre_recurrence(n)),
                ("hermite", lambda n: eqj.hermite_recurrence(n)))

    def test_weights_and_nodes_are_well_formed_at_every_order(self):
        """Positive weights, correct total mass, distinct finite nodes."""
        for name, make in self.FAMILIES:
            for n in range(2, 26):
                alpha, beta, mu0 = make(n)
                nodes, weights = eqj.gauss_quadrature(alpha, beta, mu0)
                label = "%s n=%d" % (name, n)

                self.assertTrue(bool(jnp.all(jnp.isfinite(nodes))), label)
                self.assertGreaterEqual(float(weights.min()), 0.0, label)
                self.assertAlmostEqual(float(weights.sum()), float(mu0),
                                       places=10, msg=label)
                distinct = len(np.unique(np.asarray(nodes).round(12)))
                self.assertEqual(distinct, n, label)

    def test_exactness_to_degree_two_n_minus_one(self):
        """The defining property of a Gauss rule, swept over orders.

        Checked on a random polynomial of exactly degree ``2n-1`` rather than on
        a monomial, so a rule that happened to integrate individual powers well
        could not pass by luck.
        """
        rng = np.random.default_rng(0)
        for n in range(2, 16):
            alpha, beta, mu0 = eqj.uniform_recurrence(n)
            nodes, weights = eqj.gauss_quadrature(alpha, beta, mu0)
            degree = 2 * n - 1
            coefficients = rng.normal(size=degree + 1)

            got = float(jnp.sum(weights * jnp.polyval(
                jnp.asarray(coefficients), nodes)))
            # Uniform probability measure on [-1, 1]: odd powers vanish,
            # even power p integrates to 1/(p+1).
            want = sum(c * (0.0 if (degree - i) % 2 else 1.0 / (degree - i + 1))
                       for i, c in enumerate(coefficients))
            self.assertAlmostEqual(got, want, places=8, msg="n=%d" % n)


@unittest.skipUnless(_HAS_JAX, "jax not installed")
class TestBasisInvariants(unittest.TestCase):

    def test_design_matrix_is_orthonormal_under_its_own_quadrature(self):
        """``P^T W P = I`` across dimensions and orders.

        This is what makes spectral projection exact, so it is worth checking
        everywhere rather than at one size.
        """
        for d in (1, 2, 3):
            for order in range(1, 6):
                recurrences = [eqj.uniform_recurrence(order + 2)] * d
                indices = eqj.total_order_indices(d, order)
                X, W = eqj.tensor_quadrature(recurrences)
                P = eqj.design_matrix(X, indices, recurrences)

                gram = P.T @ (W[:, None] * P)
                np.testing.assert_allclose(
                    np.array(gram), np.eye(indices.shape[0]), atol=1e-10,
                    err_msg="d=%d order=%d" % (d, order))


@unittest.skipUnless(_HAS_JAX, "jax not installed")
class TestSolverInvariants(unittest.TestCase):

    def test_lasso_solution_cannot_be_improved_by_perturbation(self):
        """Optimality, checked directly rather than by trusting convergence.

        A converged solution of a convex problem is a minimum, so no small
        perturbation may lower the objective. This checks the claim the solver
        actually makes, instead of checking that the iteration count was large.
        """
        for seed in range(4):
            rng = np.random.default_rng(seed)
            m, n = 30, 20
            A = jnp.asarray(rng.normal(size=(m, n)))
            y = jnp.asarray(rng.normal(size=m))
            lam = float(10 ** rng.uniform(-2, -0.5))

            c = eqj.lasso(A, y, lam, max_iter=20000)
            objective = lambda z: (0.5 * float(jnp.sum((A @ z - y) ** 2))
                                   + lam * float(jnp.abs(z).sum()))
            best = objective(c)

            for _ in range(100):
                perturbed = c + jnp.asarray(rng.normal(size=n)) * 1e-4
                self.assertGreaterEqual(
                    objective(perturbed), best - 1e-12,
                    "seed=%d lam=%g: a perturbation beat the solution"
                    % (seed, lam))


@unittest.skipUnless(_HAS_JAX, "jax not installed")
class TestOperatorInvariants(unittest.TestCase):

    def test_projection_and_derivative_hold_at_every_order(self):
        """Round-trip identity and exact differentiation, swept.

        The fixed-configuration tests check order 5 with one rule size. A rule
        that was marginally too small for its basis would pass there and fail
        here, which is the point of sweeping the rule size independently.
        """
        # Spread rather than exhaustive: low and high orders, a tight rule and
        # a loose one. Each configuration retraces `jacfwd`, so the full
        # 10x3 grid cost 122 s against 44 s here for the same coverage shape.
        for order in (1, 2, 3, 5, 8, 10):
            for extra in (2, 6):
                layer = eqj.SpectralOperatorLayer(
                    [eqj.uniform_recurrence(order + extra)],
                    eqj.total_order_indices(1, order))
                key = jax.random.PRNGKey(order * 10 + extra)
                c = jax.random.normal(key, (layer.n_modes, 1))
                label = "order=%d extra=%d" % (order, extra)

                np.testing.assert_allclose(
                    np.array(layer.project(layer.evaluate(c))), np.array(c),
                    atol=1e-10, err_msg=label)

                R = eqj.derivative_operator_tensor(layer)
                got = layer.integral(layer.evaluate(c), R)[:, 0]
                scalar = lambda x: (layer.design(x[None, :]) @ c)[0, 0]
                want = jax.vmap(jax.grad(scalar))(layer.X)[:, 0]
                scale = max(float(jnp.abs(want).max()), 1e-30)
                self.assertLess(float(jnp.abs(got - want).max()) / scale,
                                1e-10, label)


@unittest.skipUnless(_HAS_JAX, "jax not installed")
class TestSubspaceInvariants(unittest.TestCase):

    def test_ridge_recovery_across_random_directions_and_dimensions(self):
        """A learned subspace stays orthonormal and finds the true direction.

        The fixed test uses one direction in eight dimensions. Sweeping the
        dimension as well as the direction guards against anything that happens
        to work only at the size that was written down first.
        """
        for seed in range(3):
            rng = np.random.default_rng(seed)
            d = int(rng.integers(3, 9))
            u = rng.normal(size=(d, 1))
            u /= np.linalg.norm(u)
            X = jnp.asarray(rng.normal(size=(200, d)))
            g = lambda z: 1.0 + 2.0 * z + 0.5 * z ** 2 - 0.3 * z ** 3
            y = g(X @ jnp.asarray(u))[:, 0]

            model = eqj.PolynomialRidge(d, 1, 3)
            model.fit(X, y, steps=700, learning_rate=0.2)
            label = "seed=%d d=%d" % (seed, d)

            np.testing.assert_allclose(np.array(model.U.T @ model.U),
                                       np.eye(1), atol=1e-12, err_msg=label)
            self.assertLess(
                float(eqj.subspace_distance(model.U, jnp.asarray(u))),
                1e-6, label)


if __name__ == "__main__":
    unittest.main()
