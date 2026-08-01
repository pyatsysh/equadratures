"""Validation for Bayesian inference on the differentiable surrogate.

Acceptance gate for the inverse-problem layer: gradient-based inference must
actually *recover* something we planted, not merely run without raising.

* NUTS on an inverse-UQ problem recovers the true inputs, with the truth inside
  the 90% credible interval and the posterior sharper than the prior.
* The gradient NUTS relies on is checked directly against finite differences --
  if that were wrong the sampler would still run, just wrongly.
* The sparse-coefficient model concentrates on the true support, and its
  posterior mode agrees with the ``l1`` solve it is the Bayesian counterpart of.
* SVI reduces the ELBO loss, and ArviZ diagnostics report convergence.

Skipped unless both JAX and NumPyro are installed.
"""
import unittest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import equadratures.jax as eqj
    from equadratures.jax import inverse as eqi
    _HAS_JAX = True
    _HAS_NUMPYRO = eqi._HAS_NUMPYRO
except Exception:                                    # pragma: no cover
    _HAS_JAX = False
    _HAS_NUMPYRO = False


@unittest.skipUnless(_HAS_JAX and _HAS_NUMPYRO,
                     "jax + numpyro required (pip install equadratures[jax-bayes])")
class TestJaxInverse(unittest.TestCase):
    """Inverse UQ: recover unknown inputs from noisy observations."""

    # A forward model in three variables: two unknown inputs (x1, x2) and one
    # known control (t) that the experiment sweeps. Sweeping t is what makes
    # both unknowns identifiable from scalar measurements.
    @staticmethod
    def _truth(Z):
        x1, x2, t = Z[:, 0], Z[:, 1], Z[:, 2]
        return 1.0 + 2.0 * x1 + 0.5 * x1 ** 2 + x2 * t + 0.3 * x1 * t ** 2

    def setUp(self):
        order = 3
        self.rec = [eqj.uniform_recurrence(order + 2)] * 3
        idx = eqj.total_order_indices(3, order)
        X, W = eqj.tensor_quadrature(self.rec)
        self.poly = eqj.Poly(self.rec, idx)
        self.poly.fit_projection(X, self._truth(X), W)

        self.t_grid = jnp.linspace(-0.9, 0.9, 7)
        self.x_true = jnp.asarray([0.3, -0.5])
        self.noise = 0.02

    def _forward(self, x):
        """Surrogate output at every control setting, for inputs ``x``."""
        Z = jnp.stack([jnp.full_like(self.t_grid, x[0]),
                       jnp.full_like(self.t_grid, x[1]),
                       self.t_grid], axis=1)
        return self.poly.predict(Z)

    def _observations(self, seed=0):
        clean = self._forward(self.x_true)
        key = jax.random.PRNGKey(seed)
        return clean + self.noise * jax.random.normal(key, clean.shape)

    # ---------------------------------------------------------------- checks
    def test_inverted_prior_bounds_are_rejected_at_construction(self):
        """Fail where the mistake is, not deep inside the sampler.

        Inverted bounds do fail without this check, but they fail as "Cannot
        find valid initial parameters" from inside NUTS after a warm-up, which
        points at the model rather than at the two numbers that are the wrong
        way round. ``dist.Uniform(1, -1)`` constructs happily and returns NaN
        from ``log_prob``, so nothing earlier catches it.
        """
        forward = lambda x: jnp.asarray([x[0] + x[1], x[0] - x[1]])
        for lower, upper in (([1.0, 1.0], [-1.0, -1.0]),      # inverted
                             ([0.0, 0.0], [0.0, 0.0])):       # degenerate
            with self.assertRaises(ValueError):
                eqi.surrogate_input_model(forward, jnp.zeros(2),
                                          lower=lower, upper=upper)

    def test_surrogate_is_accurate(self):
        # The inference is only meaningful if the surrogate is exact first.
        Z = jnp.asarray(np.random.default_rng(0).uniform(-1, 1, size=(30, 3)))
        np.testing.assert_allclose(np.array(self.poly.predict(Z)),
                                   np.array(self._truth(Z)), atol=1e-10)

    def test_forward_gradient_matches_finite_differences(self):
        # This is the derivative NUTS consumes at every leapfrog step. If it is
        # wrong the sampler still runs, so it has to be checked directly.
        def scalar(x):
            return jnp.sum(self._forward(x) ** 2)

        g = np.array(jax.grad(scalar)(self.x_true))
        eps = 1e-6
        x = np.array(self.x_true)
        for i in range(len(x)):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            fd = (float(scalar(jnp.asarray(xp)))
                  - float(scalar(jnp.asarray(xm)))) / (2 * eps)
            self.assertAlmostEqual(g[i], fd, delta=1e-4 * max(1.0, abs(fd)))

    def test_nuts_recovers_true_inputs(self):
        y_obs = self._observations()
        model = eqi.surrogate_input_model(
            self._forward, y_obs, lower=[-1.0, -1.0], upper=[1.0, 1.0],
            noise_scale=self.noise)
        mcmc = eqi.run_nuts(model, num_warmup=400, num_samples=800, seed=0)

        x = np.array(eqi.posterior_samples(mcmc, "x"))
        self.assertEqual(x.shape, (800, 2))
        mean = x.mean(axis=0)
        np.testing.assert_allclose(mean, np.array(self.x_true), atol=0.05)

        # the truth sits inside the 90% credible interval, per dimension
        lo, hi = np.quantile(x, 0.05, axis=0), np.quantile(x, 0.95, axis=0)
        for i, true in enumerate(np.array(self.x_true)):
            self.assertGreaterEqual(true, lo[i])
            self.assertLessEqual(true, hi[i])

        # and the posterior is much tighter than the U(-1, 1) prior
        self.assertLess(x.std(axis=0).max(), 0.1)

    def test_nuts_infers_unknown_noise(self):
        # With noise_scale left free the sampler must also recover sigma.
        y_obs = self._observations(seed=1)
        model = eqi.surrogate_input_model(
            self._forward, y_obs, lower=[-1.0, -1.0], upper=[1.0, 1.0])
        mcmc = eqi.run_nuts(model, num_warmup=500, num_samples=800, seed=1)

        sigma = np.array(eqi.posterior_samples(mcmc, "sigma"))
        # 7 observations pin sigma only loosely, so this is a generous band
        self.assertGreater(sigma.mean(), 0.005)
        self.assertLess(sigma.mean(), 0.20)
        x = np.array(eqi.posterior_samples(mcmc, "x"))
        np.testing.assert_allclose(x.mean(axis=0), np.array(self.x_true),
                                   atol=0.1)

    def test_summarise_reports_every_site(self):
        y_obs = self._observations()
        model = eqi.surrogate_input_model(
            self._forward, y_obs, lower=[-1.0, -1.0], upper=[1.0, 1.0],
            noise_scale=self.noise)
        mcmc = eqi.run_nuts(model, num_warmup=200, num_samples=300, seed=0)
        summary = eqi.summarise(mcmc)
        self.assertIn("x", summary)
        for key in ("mean", "std", "q05", "q95"):
            self.assertIn(key, summary["x"])
        self.assertTrue(np.all(np.array(summary["x"]["q05"])
                               <= np.array(summary["x"]["q95"])))

    def test_arviz_diagnostics(self):
        y_obs = self._observations()
        model = eqi.surrogate_input_model(
            self._forward, y_obs, lower=[-1.0, -1.0], upper=[1.0, 1.0],
            noise_scale=self.noise)
        mcmc = eqi.run_nuts(model, num_warmup=300, num_samples=500,
                            num_chains=2, seed=0)
        try:
            idata = eqi.to_arviz(mcmc)
        except ImportError:                          # pragma: no cover
            self.skipTest("arviz not installed")

        import arviz as az
        rhat = az.rhat(idata)["x"].values
        self.assertTrue(np.all(rhat < 1.05),
                        "chains have not converged: r_hat = %s" % rhat)

    def test_svi_reduces_the_loss(self):
        y_obs = self._observations()
        model = eqi.surrogate_input_model(
            self._forward, y_obs, lower=[-1.0, -1.0], upper=[1.0, 1.0],
            noise_scale=self.noise)
        try:
            _, result = eqi.run_svi(model, num_steps=1500, seed=0)
        except ImportError:                          # pragma: no cover
            self.skipTest("optax not installed")
        losses = np.array(result.losses)
        self.assertTrue(np.isfinite(losses).all())
        self.assertLess(losses[-50:].mean(), losses[:50].mean())


@unittest.skipUnless(_HAS_JAX and _HAS_NUMPYRO,
                     "jax + numpyro required (pip install equadratures[jax-bayes])")
class TestJaxSparseCoefficientModel(unittest.TestCase):
    """Bayesian compressed sensing: the probabilistic sibling of the l1 solve."""

    def setUp(self):
        rng = np.random.default_rng(3)
        self.order = 4
        self.rec = [eqj.uniform_recurrence(self.order + 2)] * 2
        self.idx = eqj.total_order_indices(2, self.order)     # 15 terms
        self.X = jnp.asarray(rng.uniform(-1.0, 1.0, size=(25, 2)))
        self.A = eqj.design_matrix(self.X, self.idx, self.rec)

        # a genuinely sparse truth: three active basis terms
        c_true = np.zeros(self.idx.shape[0])
        self.support = [0, 2, 4]
        c_true[self.support] = [1.5, -2.0, 1.0]
        self.c_true = c_true
        self.y = self.A @ jnp.asarray(c_true)

    def test_posterior_concentrates_on_the_true_support(self):
        model = eqi.sparse_coefficient_model(self.A, self.y, tau_scale=0.5,
                                             noise_scale=0.01)
        mcmc = eqi.run_nuts(model, num_warmup=600, num_samples=1000, seed=0,
                            target_accept_prob=0.9)
        c = np.array(eqi.posterior_samples(mcmc, "c"))
        mean = c.mean(axis=0)

        on = np.abs(mean[self.support])
        off_idx = [i for i in range(len(mean)) if i not in self.support]
        off = np.abs(mean[off_idx])
        # active terms recovered, inactive ones shrunk far below them
        np.testing.assert_allclose(mean[self.support],
                                   self.c_true[self.support], atol=0.2)
        self.assertLess(off.max(), 0.2 * on.min())

    def test_agrees_with_the_l1_point_estimate(self):
        # The l1 solve should land inside the posterior's bulk -- they are two
        # readings of the same sparsity assumption.
        model = eqi.sparse_coefficient_model(self.A, self.y, tau_scale=0.5,
                                             noise_scale=0.01)
        mcmc = eqi.run_nuts(model, num_warmup=600, num_samples=1000, seed=0,
                            target_accept_prob=0.9)
        c_post = np.array(eqi.posterior_samples(mcmc, "c")).mean(axis=0)
        c_l1 = np.array(eqj.lasso(self.A, self.y, 1e-3, max_iter=8000))
        np.testing.assert_allclose(c_post[self.support], c_l1[self.support],
                                   atol=0.25)

    def test_uncertainty_is_reported(self):
        # The whole point over the l1 solve: every coefficient carries a spread.
        model = eqi.sparse_coefficient_model(self.A, self.y, tau_scale=0.5,
                                             noise_scale=0.01)
        mcmc = eqi.run_nuts(model, num_warmup=400, num_samples=600, seed=0,
                            target_accept_prob=0.9)
        summary = eqi.summarise(mcmc)
        std = np.array(summary["c"]["std"])
        self.assertEqual(std.shape, (self.idx.shape[0],))
        self.assertTrue(np.all(std > 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
