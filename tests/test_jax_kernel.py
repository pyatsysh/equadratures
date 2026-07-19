"""Validation for the learnable polynomial (Mercer) kernel and GP regression.

Covers the grant's D3 "random polynomial kernel": PSD by construction, usable for
GP regression, and *learnable* -- its log-spectrum trained by gradient descent
(Optax) to reduce the negative log marginal likelihood. Skipped if JAX/Optax are
absent.
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
    import optax
    _HAS_OPTAX = True
except Exception:                                    # pragma: no cover
    _HAS_OPTAX = False


def _problem(p=10, m=20):
    rec = [eqj.uniform_recurrence(p + 1)]
    idx = eqj.total_order_indices(1, p)
    kernel = eqj.PolynomialKernel(rec, idx)
    Xtr = jnp.linspace(-1.0, 1.0, m).reshape(-1, 1)
    Xte = jnp.linspace(-0.95, 0.95, 50).reshape(-1, 1)
    f = lambda X: jnp.sin(3.0 * X[:, 0])
    return kernel, Xtr, f(Xtr), Xte, f(Xte)


@unittest.skipUnless(_HAS_JAX, "jax not installed")
class TestJaxKernel(unittest.TestCase):

    def test_kernel_is_psd(self):
        kernel, Xtr, _, _, _ = _problem()
        K = np.array(kernel.gram(Xtr, Xtr, kernel.default_log_theta()))
        np.testing.assert_allclose(K, K.T, atol=1e-12)
        self.assertGreater(float(np.linalg.eigvalsh(K).min()), -1e-8)

    def test_gp_regression_fits_smooth_target(self):
        kernel, Xtr, ytr, Xte, yte = _problem()
        log_theta = kernel.default_log_theta()
        log_noise = jnp.log(1e-4)
        pred = eqj.gp_predict(kernel, Xtr, ytr, Xte, log_theta, log_noise)
        mse = float(jnp.mean((pred - yte) ** 2))
        self.assertLess(mse, 0.05)

    @unittest.skipUnless(_HAS_OPTAX, "optax not installed (pip install equadratures[jax-learn])")
    def test_learnable_kernel_training_reduces_nlml(self):
        kernel, Xtr, ytr, Xte, yte = _problem()
        params = {"log_theta": kernel.default_log_theta(),
                  "log_noise": jnp.array(jnp.log(1e-2))}

        def loss(pr):
            return eqj.gp_nlml(kernel, Xtr, ytr, pr["log_theta"], pr["log_noise"])

        def test_mse(pr):
            pred = eqj.gp_predict(kernel, Xtr, ytr, Xte, pr["log_theta"], pr["log_noise"])
            return float(jnp.mean((pred - yte) ** 2))

        nlml0, mse0 = float(loss(params)), test_mse(params)
        opt = optax.adam(5e-2)
        state = opt.init(params)
        step = jax.jit(jax.value_and_grad(loss))
        for _ in range(300):
            _, grads = step(params)
            updates, state = opt.update(grads, state, params)
            params = optax.apply_updates(params, updates)
        nlml1, mse1 = float(loss(params)), test_mse(params)

        self.assertLess(nlml1, nlml0 - 1e-3)          # training reduced the objective
        self.assertLessEqual(mse1, mse0 + 1e-9)       # generalisation did not worsen
        self.assertLess(mse1, 0.05)                   # and the fit is good


if __name__ == "__main__":
    unittest.main(verbosity=2)
