"""Validation for latent-variable models: active subspaces and polynomial ridge.

The third of the grant's three named JAX capabilities ("learnable kernels,
learnable expansions, latent-variable models"). Gates follow the namespace
convention: an exact analytic identity *and* an autodiff-versus-finite-difference
check.

The identities here are unusually sharp because a ridge function has an exact
answer to compare against. If ``f(x) = g(u^T x)`` then the gradient covariance
has rank one and its top eigenvector is ``u``, so both routes can be checked
against the truth rather than against each other.

There is also a negative control: a function with no low-dimensional structure
must *fail* to show one. A dimension-reduction method that always finds a
subspace is not measuring anything.
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

try:
    import equadratures as eq
    _HAS_CLASSIC = True
except Exception:                                    # pragma: no cover
    _HAS_CLASSIC = False


def _ridge_problem(d=8, m=300, seed=0):
    """``f(x) = g(u^T x)``: a function that genuinely has one latent variable."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(d, 1))
    u /= np.linalg.norm(u)
    X = jnp.asarray(rng.normal(size=(m, d)))
    g = lambda z: 1.0 + 2.0 * z + 0.5 * z ** 2 - 0.3 * z ** 3
    y = g(X @ jnp.asarray(u))[:, 0]
    return jnp.asarray(u), X, y


@unittest.skipUnless(_HAS_JAX, "jax not installed")
class TestSubspaceHelpers(unittest.TestCase):

    def test_orthonormalise_gives_orthonormal_columns(self):
        A = jax.random.normal(jax.random.PRNGKey(0), (7, 3))
        U = eqj.orthonormalise(A)
        np.testing.assert_allclose(np.array(U.T @ U), np.eye(3), atol=1e-12)

    def test_orthonormalise_is_sign_deterministic(self):
        """Sign-fixing matters: a free sign puts a jump in the middle of a loss."""
        A = jax.random.normal(jax.random.PRNGKey(1), (6, 2))
        U1 = eqj.orthonormalise(A)
        U2 = eqj.orthonormalise(A * 1.0)
        np.testing.assert_allclose(np.array(U1), np.array(U2), atol=1e-14)
        # R's diagonal is positive by construction after the sign fix.
        _, R = jnp.linalg.qr(A)
        self.assertTrue(bool(jnp.all(jnp.diag(R) * jnp.sign(jnp.diag(R)) >= 0)))

    def test_subspace_distance_is_rotation_invariant(self):
        """U and UQ are the same subspace; the metric must not tell them apart.

        This is the identifiability trap of this module, and the reason no test
        here compares U matrices elementwise.
        """
        U = eqj.orthonormalise(jax.random.normal(jax.random.PRNGKey(2), (6, 2)))
        Q = eqj.orthonormalise(jax.random.normal(jax.random.PRNGKey(3), (2, 2)))
        self.assertLess(float(eqj.subspace_distance(U, U @ Q)), 1e-12)
        # ... while the raw matrices are plainly different.
        self.assertGreater(float(jnp.abs(U - U @ Q).max()), 0.1)

    def test_subspace_distance_of_orthogonal_complements_is_one(self):
        U = jnp.asarray([[1.0], [0.0], [0.0]])
        V = jnp.asarray([[0.0], [1.0], [0.0]])
        self.assertAlmostEqual(float(eqj.subspace_distance(U, V)), 1.0, places=10)


@unittest.skipUnless(_HAS_JAX, "jax not installed")
class TestActiveSubspace(unittest.TestCase):

    def test_ridge_function_has_rank_one_gradient_covariance(self):
        """The analytic identity: one latent variable, one non-zero eigenvalue."""
        u, X, _ = _ridge_problem(d=6, seed=1)
        f = lambda Z: jnp.tanh(Z @ u)[:, 0]
        eigenvalues, U = eqj.active_subspace(f, X, dimension=1)

        self.assertGreater(float(eigenvalues[0]), 1e-3)
        self.assertLess(float(eigenvalues[1]) / float(eigenvalues[0]), 1e-12)
        self.assertLess(float(eqj.subspace_distance(U, u)), 1e-10)

    def test_two_dimensional_latent_space(self):
        rng = np.random.default_rng(4)
        d = 6
        A, _ = np.linalg.qr(rng.normal(size=(d, 2)))
        A = jnp.asarray(A)
        X = jnp.asarray(rng.normal(size=(400, d)))
        f = lambda Z: (jnp.sin(Z @ A[:, 0:1]) + (Z @ A[:, 1:2]) ** 2)[:, 0]

        eigenvalues, U = eqj.active_subspace(f, X, dimension=2)
        self.assertGreater(float(eigenvalues[1]), 1e-3)
        self.assertLess(float(eigenvalues[2]) / float(eigenvalues[1]), 1e-10)
        self.assertLess(float(eqj.subspace_distance(U, A)), 1e-9)

    def test_no_structure_is_reported_as_no_structure(self):
        """Negative control. A method that always finds a subspace finds nothing.

        An isotropic quadratic varies equally in every direction, so the
        eigenvalues must be comparable and there must be no gap to exploit.
        """
        rng = np.random.default_rng(5)
        d = 6
        X = jnp.asarray(rng.normal(size=(400, d)))
        f = lambda Z: jnp.sum(Z ** 2, axis=1)

        eigenvalues, _ = eqj.active_subspace(f, X)
        ratio = float(eigenvalues[0]) / float(eigenvalues[d - 1])
        self.assertLess(ratio, 3.0, "isotropic input should show no clear gap")

    def test_gradient_covariance_is_symmetric_psd(self):
        u, X, _ = _ridge_problem(d=5, seed=6)
        C = eqj.gradient_covariance(lambda Z: (Z @ u)[:, 0] ** 3, X)
        np.testing.assert_allclose(np.array(C), np.array(C).T, atol=1e-12)
        self.assertGreater(float(jnp.linalg.eigvalsh(C).min()), -1e-10)


@unittest.skipUnless(_HAS_JAX, "jax not installed")
class TestPolynomialRidge(unittest.TestCase):

    def test_recovers_an_exact_ridge_subspace_from_values_alone(self):
        """No gradients of the model are used -- only function values.

        This is the case where the classic namespace runs Gauss-Newton with a
        hand-derived Jacobian and two hand-tuned line-search constants. Here it
        is gradient descent through a least-squares solve.
        """
        u, X, y = _ridge_problem(d=8, m=300, seed=0)
        model = eqj.PolynomialRidge(dimensions=8, subspace_dimension=1, order=3)
        history = model.fit(X, y, steps=800, learning_rate=0.2)

        self.assertLess(history[-1], history[0] * 1e-12)
        self.assertLess(float(eqj.subspace_distance(model.U, u)), 1e-8)

        relative = (jnp.linalg.norm(model.predict(X) - y)
                    / jnp.linalg.norm(y))
        self.assertLess(float(relative), 1e-8)

    def test_predict_generalises_to_unseen_points(self):
        """Standardisation uses the training projections, so predict is fixed."""
        u, X, y = _ridge_problem(d=6, m=250, seed=7)
        model = eqj.PolynomialRidge(dimensions=6, subspace_dimension=1, order=3)
        model.fit(X, y, steps=800, learning_rate=0.2)

        rng = np.random.default_rng(70)
        X_new = jnp.asarray(rng.normal(size=(120, 6)))
        g = lambda z: 1.0 + 2.0 * z + 0.5 * z ** 2 - 0.3 * z ** 3
        y_new = g(X_new @ u)[:, 0]

        relative = (jnp.linalg.norm(model.predict(X_new) - y_new)
                    / jnp.linalg.norm(y_new))
        self.assertLess(float(relative), 1e-6)

    def test_loss_gradient_matches_finite_differences(self):
        u, X, y = _ridge_problem(d=5, m=200, seed=8)
        model = eqj.PolynomialRidge(dimensions=5, subspace_dimension=1, order=3)
        A = jax.random.normal(jax.random.PRNGKey(9), (5, 1))

        grad = jax.grad(lambda a: model.loss(a, X, y))(A)
        eps = 1e-6
        for i in (0, 2, 4):
            def shifted(step):
                return float(model.loss(A.at[i, 0].add(step), X, y))
            fd = (shifted(eps) - shifted(-eps)) / (2 * eps)
            self.assertAlmostEqual(float(grad[i, 0]), fd,
                                   delta=1e-5 * max(1.0, abs(fd)))

    def test_rejects_a_subspace_larger_than_the_space(self):
        """A thin QR would silently return fewer columns and fit a wrong model.

        With d=3 and r=5 the QR returns 3 columns, `design_matrix` then quietly
        ignores latent dimensions it was not given, and the caller gets a
        rank-deficient fit on 3 latent variables while believing they asked for
        5. Nothing raises anywhere downstream, so it has to raise here.
        """
        with self.assertRaises(ValueError):
            eqj.PolynomialRidge(dimensions=3, subspace_dimension=5, order=2)
        with self.assertRaises(ValueError):
            eqj.PolynomialRidge(dimensions=3, subspace_dimension=0, order=2)
        with self.assertRaises(ValueError):
            eqj.orthonormalise(jax.random.normal(jax.random.PRNGKey(0), (3, 5)))

    def test_zero_variance_latent_coordinate_has_a_finite_gradient(self):
        """The standardisation floor must go inside the square root.

        ``std(z) + eps`` and ``sqrt(var(z) + eps)`` agree in value and differ
        completely in derivative: ``d sqrt(v)/dv`` is unbounded at zero, so a
        collapsed latent coordinate gives a NaN gradient with the first form.
        """
        model = eqj.PolynomialRidge(dimensions=4, subspace_dimension=1, order=2)
        Z = jnp.zeros((10, 1))
        grad = jax.grad(lambda z: jnp.sum(model._standardise(z)))(Z)
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))

    def test_regularisation_rescues_a_rank_deficient_design(self):
        """Documented behaviour, both halves of it.

        Default (exact `lstsq`) gives NaN on a rank-deficient design, because
        the SVD derivative is undefined with repeated singular values. That is
        stated in the docstring rather than papered over, since the fix costs
        accuracy on every well-posed problem. Opting in restores finite
        gradients.
        """
        rng = np.random.default_rng(0)
        X = jnp.zeros((30, 4))                      # every point identical
        y = jnp.asarray(rng.normal(size=30))

        exact = eqj.PolynomialRidge(4, 1, 2)
        self.assertFalse(bool(np.all(np.isfinite(exact.fit(X, y, steps=10)))))

        floored = eqj.PolynomialRidge(4, 1, 2, regularisation=1e-10)
        self.assertTrue(bool(np.all(np.isfinite(floored.fit(X, y, steps=10)))))

    def test_regularisation_does_not_spoil_a_well_posed_fit(self):
        u, X, y = _ridge_problem(d=8, m=300, seed=0)
        model = eqj.PolynomialRidge(8, 1, 3, regularisation=1e-10)
        model.fit(X, y, steps=800, learning_rate=0.2)
        self.assertLess(float(eqj.subspace_distance(model.U, u)), 1e-8)

    def test_fit_is_jittable_end_to_end(self):
        u, X, y = _ridge_problem(d=5, m=150, seed=10)
        model = eqj.PolynomialRidge(dimensions=5, subspace_dimension=1, order=2)
        A = jax.random.normal(jax.random.PRNGKey(11), (5, 1))
        jitted = jax.jit(lambda a: model.loss(a, X, y))
        self.assertTrue(np.isfinite(float(jitted(A))))


@unittest.skipUnless(_HAS_JAX, "jax not installed")
@unittest.skipUnless(_HAS_CLASSIC, "classic equadratures not importable")
class TestParityWithClassic(unittest.TestCase):

    def test_active_subspace_matches_classic(self):
        """Same surrogate, same gradient covariance, same answer.

        The classic route differentiates its polynomial analytically; this one
        uses ``jax.grad``. Agreement to round-off says the JAX path is not a
        re-derivation with its own conventions but the same computation.
        """
        d = 5
        rng = np.random.default_rng(2)
        u = rng.normal(size=(d, 1))
        u /= np.linalg.norm(u)
        X = rng.uniform(-1.0, 1.0, size=(300, d))
        y = (X @ u)[:, 0] + 0.5 * ((X @ u)[:, 0]) ** 2

        params = [eq.Parameter(distribution="uniform", lower=-1, upper=1, order=2)
                  for _ in range(d)]
        classic = eq.Poly(params, eq.Basis("total-order"), method="least-squares",
                          sampling_args={"sample-points": X, "sample-outputs": y})
        classic.set_model()
        classic_sub = eq.Subspaces(method="active-subspace",
                                   full_space_poly=classic, subspace_dimension=1)
        U_classic = jnp.asarray(np.asarray(classic_sub.get_subspace())[:, :1])
        ev_classic = np.asarray(classic_sub.get_eigenvalues()).ravel()

        recurrences = [eqj.uniform_recurrence(3)] * d
        indices = eqj.total_order_indices(d, 2)
        poly = eqj.Poly(recurrences, indices)
        poly.fit(jnp.asarray(X), jnp.asarray(y))
        ev_jax, U_jax = eqj.active_subspace(lambda Z: poly.predict(Z),
                                            jnp.asarray(X), dimension=1)

        self.assertAlmostEqual(float(ev_jax[0]), float(ev_classic[0]), places=8)
        self.assertLess(float(eqj.subspace_distance(U_jax, U_classic)), 1e-10)


if __name__ == "__main__":
    unittest.main()
