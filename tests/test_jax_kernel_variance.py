"""Validation for GP predictive variance on the polynomial kernel.

``gp_predict`` returns only a posterior mean; without a variance the model is
kernel ridge regression wearing a GP's clothes. These tests pin down the four
things the variance has to get right:

* it agrees with ``gp_predict`` on the mean (same algebra, different route);
* it matches a direct, independent evaluation of
  ``k(x*,x*) - k(x*,X)[K + s^2 I]^-1 k(X,x*)``;
* it is non-negative, shrinks at the training points, and grows away from them;
* ``include_noise`` adds exactly the observation variance;
* it is differentiable, since calibration terms in a training loss depend on it.

One deliberate check encodes a *limitation*: the kernel has finite rank, so once
the data pin down every polynomial feature the variance stays bounded rather
than reverting to a prior. Users need to know that before reading the error bars
as model-form uncertainty.

NOTE: written while this machine was under a compute lease and therefore
**never executed**. It is queued in the deferred verification pass.
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


def _setup(p=6, m=12):
    rec = [eqj.uniform_recurrence(p + 1)]
    idx = eqj.total_order_indices(1, p)
    kernel = eqj.PolynomialKernel(rec, idx)
    Xtr = jnp.linspace(-1.0, 1.0, m).reshape(-1, 1)
    ytr = jnp.sin(3.0 * Xtr[:, 0])
    log_theta = kernel.default_log_theta()
    log_noise = jnp.log(jnp.asarray(1e-4))            # log of the *variance*
    return kernel, Xtr, ytr, log_theta, log_noise


@unittest.skipUnless(_HAS_JAX, "jax not installed (pip install equadratures[jax])")
class TestGPPredictiveVariance(unittest.TestCase):

    def test_mean_agrees_with_gp_predict(self):
        kernel, Xtr, ytr, lt, ln = _setup()
        Xte = jnp.linspace(-0.9, 0.9, 25).reshape(-1, 1)
        mean_a = eqj.gp_predict(kernel, Xtr, ytr, Xte, lt, ln)
        mean_b, _ = eqj.gp_predict_with_variance(kernel, Xtr, ytr, Xte, lt, ln)
        np.testing.assert_allclose(np.array(mean_b), np.array(mean_a), atol=1e-9)

    def test_variance_matches_direct_formula(self):
        # Independent evaluation of the textbook expression, via an explicit
        # inverse rather than the Cholesky path the implementation uses.
        kernel, Xtr, ytr, lt, ln = _setup()
        Xte = jnp.linspace(-0.9, 0.9, 15).reshape(-1, 1)

        m = Xtr.shape[0]
        K = np.array(kernel.gram(Xtr, Xtr, lt)) + float(jnp.exp(ln)) * np.eye(m)
        Ks = np.array(kernel.gram(Xte, Xtr, lt))
        Kss = np.array(kernel.gram(Xte, Xte, lt))
        expected = np.diag(Kss) - np.einsum("ij,jk,ik->i", Ks,
                                            np.linalg.inv(K), Ks)

        _, var = eqj.gp_predict_with_variance(kernel, Xtr, ytr, Xte, lt, ln)
        np.testing.assert_allclose(np.array(var), expected, rtol=1e-6, atol=1e-9)

    def test_variance_is_non_negative(self):
        kernel, Xtr, ytr, lt, ln = _setup()
        Xte = jnp.linspace(-1.0, 1.0, 200).reshape(-1, 1)
        _, var = eqj.gp_predict_with_variance(kernel, Xtr, ytr, Xte, lt, ln)
        self.assertTrue(bool(jnp.all(var >= 0.0)))

    def test_variance_is_smaller_at_the_data(self):
        # Uncertainty should be lowest where we have observations. Use a sparse
        # training set so there are genuine gaps to be uncertain about.
        rec = [eqj.uniform_recurrence(7)]
        idx = eqj.total_order_indices(1, 6)
        kernel = eqj.PolynomialKernel(rec, idx)
        Xtr = jnp.asarray([[-0.9], [-0.5], [0.5], [0.9]])
        ytr = jnp.sin(3.0 * Xtr[:, 0])
        lt, ln = kernel.default_log_theta(), jnp.log(jnp.asarray(1e-6))

        _, var_at = eqj.gp_predict_with_variance(kernel, Xtr, ytr, Xtr, lt, ln)
        gap = jnp.asarray([[0.0]])                     # the middle of the gap
        _, var_gap = eqj.gp_predict_with_variance(kernel, Xtr, ytr, gap, lt, ln)
        self.assertLess(float(jnp.max(var_at)), float(var_gap[0]))

    def test_include_noise_adds_exactly_the_noise_variance(self):
        kernel, Xtr, ytr, lt, ln = _setup()
        Xte = jnp.linspace(-0.8, 0.8, 10).reshape(-1, 1)
        _, latent = eqj.gp_predict_with_variance(kernel, Xtr, ytr, Xte, lt, ln)
        _, noisy = eqj.gp_predict_with_variance(kernel, Xtr, ytr, Xte, lt, ln,
                                                include_noise=True)
        np.testing.assert_allclose(np.array(noisy - latent),
                                   np.full(Xte.shape[0], float(jnp.exp(ln))),
                                   rtol=1e-9, atol=1e-12)

    def test_variance_grows_with_the_noise_level(self):
        kernel, Xtr, ytr, lt, _ = _setup()
        Xte = jnp.asarray([[0.13]])
        v_small = eqj.gp_predict_with_variance(
            kernel, Xtr, ytr, Xte, lt, jnp.log(jnp.asarray(1e-6)))[1]
        v_large = eqj.gp_predict_with_variance(
            kernel, Xtr, ytr, Xte, lt, jnp.log(jnp.asarray(1e-2)))[1]
        self.assertGreater(float(v_large[0]), float(v_small[0]))

    def test_variance_is_differentiable(self):
        # Calibration-aware training objectives differentiate the variance, so
        # this has to work and be finite.
        kernel, Xtr, ytr, _, _ = _setup()
        Xte = jnp.linspace(-0.8, 0.8, 12).reshape(-1, 1)

        def total_variance(log_theta):
            _, var = eqj.gp_predict_with_variance(
                kernel, Xtr, ytr, Xte, log_theta, jnp.log(jnp.asarray(1e-4)))
            return jnp.sum(var)

        lt0 = kernel.default_log_theta()
        g = np.array(jax.grad(total_variance)(lt0))
        self.assertEqual(g.shape, (kernel.n_features,))
        self.assertTrue(np.all(np.isfinite(g)))

        # spot-check one component against a finite difference
        eps, i = 1e-5, 2
        lp = np.array(lt0); lp[i] += eps
        lm = np.array(lt0); lm[i] -= eps
        fd = (float(total_variance(jnp.asarray(lp)))
              - float(total_variance(jnp.asarray(lm)))) / (2 * eps)
        self.assertAlmostEqual(g[i], fd, delta=1e-4 * max(1.0, abs(fd)))

    def test_finite_rank_keeps_variance_bounded(self):
        # A documented limitation, asserted so it cannot regress silently into a
        # claim the model does not support: with more training points than
        # features, the posterior variance is small *everywhere*, including far
        # outside the data. These error bars describe uncertainty within the
        # polynomial span, not model-form error.
        rec = [eqj.uniform_recurrence(5)]
        idx = eqj.total_order_indices(1, 4)            # 5 features
        kernel = eqj.PolynomialKernel(rec, idx)
        Xtr = jnp.linspace(-1.0, 1.0, 40).reshape(-1, 1)
        ytr = jnp.sin(2.0 * Xtr[:, 0])
        lt, ln = kernel.default_log_theta(), jnp.log(jnp.asarray(1e-8))

        Xte = jnp.asarray([[0.0], [0.999]])
        _, var = eqj.gp_predict_with_variance(kernel, Xtr, ytr, Xte, lt, ln)
        k_self = np.diag(np.array(kernel.gram(Xte, Xte, lt)))
        # variance collapses to a small fraction of the prior k(x*, x*)
        self.assertTrue(np.all(np.array(var) < 0.05 * k_self))


if __name__ == "__main__":
    unittest.main(verbosity=2)
