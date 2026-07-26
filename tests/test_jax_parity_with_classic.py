"""Parity of ``equadratures.jax`` against the classic NumPy ``equadratures``.

The JAX namespace is a re-implementation, not a wrapper, so "does it agree with
the library it is meant to be a backend for?" is the first question any reviewer
will ask. This module answers it by comparing the two implementations directly,
at every level of the stack:

* recurrence coefficients (analytic families *and* the Stieltjes procedure on an
  arbitrary discretised measure),
* Gauss points and weights, in one dimension and on a tensor grid,
* the multivariate orthonormal design matrix,
* fitted coefficients,
* the UQ outputs -- mean, variance, first-order and total Sobol' indices,
* surrogate predictions at unseen points.

Conventions differ in two places and the tests handle both explicitly rather
than papering over them:

1. Classic stores the *monic* recurrence, whose off-diagonals are the squares of
   the symmetric Jacobi off-diagonals used here, with ``ab[0, 1]`` holding
   ``mu0``. So ``jax_beta == sqrt(classic_ab[1:, 1])``.
2. The classic tensor-grid basis enumerates multi-indices in a different order
   from :func:`equadratures.jax.basis.tensor_grid_indices`, so coefficient
   vectors are compared after aligning on the multi-indices themselves. (For a
   *total-order* basis the two orderings already coincide, and the test asserts
   that.)

Skipped if JAX is absent.
"""
import unittest
import numpy as np

try:
    import jax.numpy as jnp
    import equadratures as eq
    from equadratures.distributions.recurrence_utils import (
        custom_recurrence_coefficients,
    )
    import equadratures.jax as eqj
    _HAS_JAX = True
except Exception:                                    # pragma: no cover
    _HAS_JAX = False


def _align(index_from, index_to):
    """Row permutation taking ``index_from`` order to ``index_to`` order."""
    lookup = {tuple(int(v) for v in row): i for i, row in enumerate(index_from)}
    return np.array([lookup[tuple(int(v) for v in row)] for row in index_to])


@unittest.skipUnless(_HAS_JAX, "jax not installed (pip install equadratures[jax])")
class TestJaxParityWithClassic(unittest.TestCase):

    # ------------------------------------------------------------------ setup
    def _classic_uniform(self, order):
        return eq.Parameter(distribution="uniform", lower=-1.0, upper=1.0,
                            order=order)

    def _f(self, Z):
        """Test model, exactly representable in a degree-2 basis."""
        return 1.0 + 2.0 * Z[:, 0] + 3.0 * Z[:, 0] * Z[:, 1]

    # ------------------------------------------------------- recurrence layer
    def test_uniform_recurrence_matches_classic(self):
        p = self._classic_uniform(6)
        ab = np.array(p.get_recurrence_coefficients(6))
        alpha, beta, mu0 = eqj.uniform_recurrence(ab.shape[0])
        np.testing.assert_allclose(np.array(alpha), ab[:, 0], atol=1e-12)
        np.testing.assert_allclose(np.array(beta), np.sqrt(ab[1:, 1]), atol=1e-12)
        self.assertAlmostEqual(float(mu0), float(ab[0, 1]), places=12)

    def test_gaussian_recurrence_matches_classic(self):
        p = eq.Parameter(distribution="gaussian", shape_parameter_A=0.0,
                         shape_parameter_B=1.0, order=6)
        ab = np.array(p.get_recurrence_coefficients(6))
        alpha, beta, mu0 = eqj.hermite_recurrence(ab.shape[0])
        np.testing.assert_allclose(np.array(alpha), ab[:, 0], atol=1e-10)
        np.testing.assert_allclose(np.array(beta), np.sqrt(ab[1:, 1]), atol=1e-10)
        self.assertAlmostEqual(float(mu0), float(ab[0, 1]), places=10)

    def test_stieltjes_matches_classic_on_arbitrary_distribution(self):
        # The strongest recurrence check: hand both implementations the *same*
        # discretised measure (a Beta(2,3) density on classic's own pdf grid)
        # and require identical coefficients. This is what licenses the JAX
        # pipeline on distributions with no analytic recurrence.
        # Shape parameters are kept >= 1 so the density stays bounded. Below 1
        # the Beta density diverges at the endpoints and classic's equispaced
        # pdf grid evaluates to inf there, which no discretised Stieltjes
        # procedure -- ours or theirs -- can recover from. That is a property of
        # the discretisation, not of either implementation.
        for a_shape, b_shape in ((2.0, 3.0), (1.5, 4.0), (5.0, 1.5)):
            p = eq.Parameter(distribution="beta", shape_parameter_A=a_shape,
                             shape_parameter_B=b_shape, lower=0.0, upper=1.0,
                             order=5)
            x = np.array(p.x_range_for_pdf)
            # Classic normalises the density to unit mass internally; feed our
            # (deliberately more general, mass-preserving) port the same
            # normalised measure so mu0 is comparable. alpha and the monic beta
            # are invariant under this rescaling anyway.
            w = np.array(p.get_pdf(x))
            w = w / w.sum()
            ab = np.array(custom_recurrence_coefficients(x, w, 5))

            n = ab.shape[0]
            alpha, beta, mu0 = eqj.stieltjes_recurrence(jnp.asarray(x),
                                                        jnp.asarray(w), n)
            np.testing.assert_allclose(np.array(alpha), ab[:, 0], rtol=1e-10,
                                       atol=1e-12)
            np.testing.assert_allclose(np.array(beta), np.sqrt(ab[1:, 1]),
                                       rtol=1e-10, atol=1e-12)
            self.assertAlmostEqual(float(mu0), float(ab[0, 1]), places=10)

    # ------------------------------------------------------- quadrature layer
    def test_1d_points_and_weights_match_classic(self):
        order = 5
        p = self._classic_uniform(order)
        basis = eq.Basis("tensor-grid", orders=[order])
        poly = eq.Poly([p], basis, method="numerical-integration")
        xc, wc = poly.get_points_and_weights()

        alpha, beta, mu0 = eqj.uniform_recurrence(order + 1)
        xj, wj = eqj.gauss_quadrature(alpha, beta, mu0)

        order_c = np.argsort(xc.ravel())
        np.testing.assert_allclose(xc.ravel()[order_c], np.array(xj), atol=1e-12)
        np.testing.assert_allclose(wc.ravel()[order_c], np.array(wj), atol=1e-12)

    def test_tensor_grid_points_and_weights_match_classic(self):
        order = 4
        p = self._classic_uniform(order)
        basis = eq.Basis("tensor-grid", orders=[order, order])
        poly = eq.Poly([p, p], basis, method="numerical-integration")
        xc, wc = poly.get_points_and_weights()

        Xj, Wj = eqj.tensor_quadrature([eqj.uniform_recurrence(order + 1)] * 2)
        Xj, Wj = np.array(Xj), np.array(Wj)

        oc = np.lexsort((xc[:, 1], xc[:, 0]))
        oj = np.lexsort((Xj[:, 1], Xj[:, 0]))
        np.testing.assert_allclose(xc[oc], Xj[oj], atol=1e-12)
        np.testing.assert_allclose(wc.ravel()[oc], Wj[oj], atol=1e-12)

    # ------------------------------------------------------------ basis layer
    def test_total_order_multi_index_ordering_matches_classic(self):
        for dim, order in ((2, 3), (2, 5), (3, 3)):
            basis = eq.Basis("total-order", orders=[order] * dim)
            p = self._classic_uniform(order)
            poly = eq.Poly([p] * dim, basis, method="least-squares")
            mi = np.array(poly.get_multi_index()).astype(int)
            np.testing.assert_array_equal(mi, eqj.total_order_indices(dim, order))

    def test_design_matrix_matches_classic(self):
        order = 3
        p = self._classic_uniform(order)
        basis = eq.Basis("total-order", orders=[order, order])
        poly = eq.Poly([p, p], basis, method="least-squares")

        rng = np.random.default_rng(0)
        X = rng.uniform(-1.0, 1.0, size=(12, 2))
        P = np.array(poly.get_poly(X))                     # (n_basis, n_points)

        A = np.array(eqj.design_matrix(
            jnp.asarray(X), eqj.total_order_indices(2, order),
            [eqj.uniform_recurrence(order + 2)] * 2))      # (n_points, n_basis)
        np.testing.assert_allclose(A, P.T, atol=1e-11)

    # ------------------------------------------------ coefficients and output
    def _classic_projection(self, order):
        p = self._classic_uniform(order)
        basis = eq.Basis("tensor-grid", orders=[order, order])
        poly = eq.Poly([p, p], basis, method="numerical-integration")
        xc, _ = poly.get_points_and_weights()
        poly.set_model(self._f(xc).reshape(-1, 1))
        return poly

    def _jax_projection(self, order):
        rec = [eqj.uniform_recurrence(order + 1)] * 2
        idx = eqj.tensor_grid_indices(2, order)
        X, W = eqj.tensor_quadrature(rec)
        poly = eqj.Poly(rec, idx)
        poly.fit_projection(X, self._f(X), W)
        return poly, idx

    def test_coefficients_match_classic(self):
        order = 4
        pc = self._classic_projection(order)
        pj, idx_j = self._jax_projection(order)

        idx_c = np.array(pc.get_multi_index()).astype(int)
        perm = _align(idx_c, idx_j)
        cc = np.array(pc.get_coefficients()).ravel()[perm]
        np.testing.assert_allclose(np.array(pj.coefficients), cc, atol=1e-10)

    def test_mean_and_variance_match_classic(self):
        order = 4
        pc = self._classic_projection(order)
        pj, _ = self._jax_projection(order)
        mc, vc = pc.get_mean_and_variance()
        self.assertAlmostEqual(float(pj.mean()), float(mc), places=10)
        self.assertAlmostEqual(float(pj.variance()), float(vc), places=10)
        # ... and both agree with the analytic answer
        self.assertAlmostEqual(float(mc), 1.0, places=10)
        self.assertAlmostEqual(float(vc), 7.0 / 3.0, places=10)

    def test_sobol_indices_match_classic(self):
        order = 4
        pc = self._classic_projection(order)
        pj, _ = self._jax_projection(order)

        first_c = pc.get_sobol_indices(order=1)
        Sc = np.array([first_c[(0,)], first_c[(1,)]])
        Tc = np.array(pc.get_total_sobol_indices()).ravel()

        np.testing.assert_allclose(np.array(pj.sobol_indices()), Sc, atol=1e-10)
        np.testing.assert_allclose(np.array(pj.total_sobol_indices()), Tc,
                                   atol=1e-10)
        np.testing.assert_allclose(Sc, [4.0 / 7.0, 0.0], atol=1e-10)

    def test_predictions_match_classic(self):
        order = 4
        pc = self._classic_projection(order)
        pj, _ = self._jax_projection(order)

        rng = np.random.default_rng(1)
        Xt = rng.uniform(-1.0, 1.0, size=(20, 2))
        yc = np.array(pc.get_polyfit(Xt)).ravel()
        yj = np.array(pj.predict(jnp.asarray(Xt)))
        np.testing.assert_allclose(yj, yc, atol=1e-10)
        np.testing.assert_allclose(yj, self._f(Xt), atol=1e-10)

    def test_least_squares_fit_matches_classic(self):
        # Same sample points, same basis, independent solvers.
        order = 3
        p = self._classic_uniform(order)
        basis = eq.Basis("total-order", orders=[order, order])
        rng = np.random.default_rng(2)
        X = rng.uniform(-1.0, 1.0, size=(40, 2))
        y = self._f(X)

        poly_c = eq.Poly([p, p], basis, method="least-squares",
                         sampling_args={"mesh": "user-defined",
                                        "sample-points": X,
                                        "sample-outputs": y.reshape(-1, 1)})
        poly_c.set_model()

        rec = [eqj.uniform_recurrence(order + 2)] * 2
        poly_j = eqj.Poly(rec, eqj.total_order_indices(2, order))
        poly_j.fit(jnp.asarray(X), jnp.asarray(y))

        cc = np.array(poly_c.get_coefficients()).ravel()
        np.testing.assert_allclose(np.array(poly_j.coefficients), cc, atol=1e-9)

        mc, vc = poly_c.get_mean_and_variance()
        self.assertAlmostEqual(float(poly_j.mean()), float(mc), places=9)
        self.assertAlmostEqual(float(poly_j.variance()), float(vc), places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
