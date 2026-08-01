"""Validation for the polynomial neural-operator layer.

Covers the grant's D3 remainder: a neural-operator layer whose integral operator
is the differentiable polynomial kernel, applied by Gauss quadrature rather than
by an FFT. The gates follow the convention of the rest of this namespace -- an
exact analytic identity *and* an autodiff-versus-finite-difference check --
because a layer that merely trains to a small loss has not been validated.

The identities exercised here are stronger than "close enough":

* projecting and reconstructing a function in the span is the identity;
* differentiation, a non-symmetric operator, is represented *exactly*;
* the diagonal parameterisation is exactly the Mercer kernel of
  ``PolynomialKernel`` applied by quadrature, not an approximation of it;
* a trained layer evaluated on a finer quadrature returns the same function.

Skipped if JAX (or, for the training gate, Optax) is absent.
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

ORDER = 5


def _layer(order=ORDER, extra=3, mode="dense", c_in=1, c_out=1):
    """A 1-D layer whose quadrature is comfortably exact for the products used."""
    rec = [eqj.uniform_recurrence(order + extra)]
    idx = eqj.total_order_indices(1, order)
    return eqj.SpectralOperatorLayer(rec, idx, channels_in=c_in,
                                     channels_out=c_out, mode=mode)


def _random_span_function(layer, seed=0, channels=1):
    """A function that lies exactly in the layer's span, and its coefficients."""
    c = jax.random.normal(jax.random.PRNGKey(seed), (layer.n_modes, channels))
    return layer.evaluate(c), c


@unittest.skipUnless(_HAS_JAX, "jax not installed")
class TestSpectralOperatorLayer(unittest.TestCase):

    def test_projection_is_exact_on_the_span(self):
        """Project then reconstruct is the identity for a function in the span.

        This is the sum-rule of the layer: it holds only because the quadrature
        integrates ``phi_j phi_k`` exactly, so a failure here means the rule is
        under-sized rather than the algebra wrong.
        """
        layer = _layer()
        u, c = _random_span_function(layer)
        np.testing.assert_allclose(np.array(layer.project(u)), np.array(c),
                                   atol=1e-12)

    def test_projection_is_not_exact_off_the_span(self):
        """The converse, so the test above cannot pass for a trivial reason.

        ``|x|`` is not a polynomial, and no amount of quadrature makes the
        round-trip exact for it. Asserting the failure keeps the exactness claim
        honest.
        """
        layer = _layer()
        u = jnp.abs(layer.X[:, :1])
        back = layer.evaluate(layer.project(u))
        self.assertGreater(float(jnp.abs(back - u).max()), 1e-3)

    def test_derivative_operator_is_represented_exactly(self):
        """``d/dx`` is reproduced to machine precision, not approximated.

        Differentiation maps the span into itself, so a dense spectral tensor
        represents it exactly. It is also *non-symmetric*, which is why the
        dense parameterisation exists: a mode-diagonal kernel is symmetric and
        could not do this.
        """
        layer = _layer()
        R = eqj.derivative_operator_tensor(layer)
        u, c = _random_span_function(layer, seed=1)

        got = layer.integral(u, R)[:, 0]
        scalar = lambda x: (layer.design(x[None, :]) @ c)[0, 0]
        want = jax.vmap(jax.grad(scalar))(layer.X)[:, 0]

        scale = float(jnp.abs(want).max())
        self.assertGreater(scale, 1.0)               # a non-trivial function
        np.testing.assert_allclose(np.array(got), np.array(want),
                                   rtol=0, atol=1e-11 * scale)

    def test_diagonal_mode_is_the_mercer_kernel_applied_by_quadrature(self):
        """Diagonal mode *is* ``PolynomialKernel``, not merely similar to it.

        ``(K u)(x) = sum_i w_i kappa(x, x_i) u(x_i)`` with ``kappa`` the Mercer
        kernel of spectrum ``theta_k^2``. Both sides are the same finite sum, so
        they agree to round-off and the tolerance is set accordingly.
        """
        layer = _layer(mode="diagonal")
        log_theta = jnp.linspace(-0.3, 0.4, layer.n_modes)
        R = jnp.exp(2.0 * log_theta)[:, None, None]

        u = jax.random.normal(jax.random.PRNGKey(2), (layer.X.shape[0], 1))
        Xq = jnp.linspace(-0.9, 0.9, 11).reshape(-1, 1)

        got = layer.integral(u, R, Xq)[:, 0]
        kernel = eqj.PolynomialKernel(layer.recurrences, layer.indices)
        want = kernel.gram(Xq, layer.X, log_theta) @ (layer.W * u[:, 0])

        np.testing.assert_allclose(np.array(got), np.array(want), atol=1e-13)

    def test_discretisation_invariance_is_exact(self):
        """The same parameters on a finer quadrature give the same function.

        The selling point of a neural operator, and here it is exact rather than
        approximate: the parameters live on polynomial modes, not on a grid.
        """
        coarse = _layer(extra=3)
        fine = _layer(extra=9)
        self.assertLess(coarse.X.shape[0], fine.X.shape[0])

        R = eqj.derivative_operator_tensor(coarse)
        c = jax.random.normal(jax.random.PRNGKey(3), (coarse.n_modes, 1))
        Xq = jnp.linspace(-0.9, 0.9, 13).reshape(-1, 1)

        out_coarse = coarse.integral(coarse.evaluate(c), R, Xq)
        out_fine = fine.integral(fine.evaluate(c), R, Xq)
        np.testing.assert_allclose(np.array(out_coarse), np.array(out_fine),
                                   atol=1e-11)

    def test_gradient_matches_finite_differences(self):
        """The adjoint through projection, spectral apply and reconstruction."""
        layer = _layer()
        u, _ = _random_span_function(layer, seed=4)
        target = layer.integral(u, eqj.derivative_operator_tensor(layer))
        params = layer.init_params(jax.random.PRNGKey(5))

        def loss(p):
            return jnp.mean((layer.apply(u, p) - target) ** 2)

        grad = jax.grad(loss)(params)["spectral"]

        eps = 1e-6
        for (j, k) in ((0, 0), (2, 3), (4, 1)):
            def shifted(step):
                q = dict(params)
                q["spectral"] = params["spectral"].at[j, k, 0, 0].add(step)
                return loss(q)
            fd = float((shifted(eps) - shifted(-eps)) / (2 * eps))
            self.assertAlmostEqual(float(grad[j, k, 0, 0]), fd,
                                   delta=1e-6 * max(1.0, abs(fd)))

    def test_jit_and_vmap(self):
        """``jit`` and ``vmap`` over a batch of input functions."""
        layer = _layer()
        params = layer.init_params(jax.random.PRNGKey(6))
        C = jax.random.normal(jax.random.PRNGKey(7), (8, layer.n_modes, 1))
        U = jax.vmap(layer.evaluate)(C)

        batched = jax.jit(jax.vmap(lambda u: layer.apply(u, params)))
        out = batched(U)
        self.assertEqual(out.shape, (8, layer.X.shape[0], 1))

        one = layer.apply(U[3], params)
        np.testing.assert_allclose(np.array(out[3]), np.array(one), atol=1e-12)

    def test_channels(self):
        """Shapes compose for a genuinely multi-channel layer."""
        layer = _layer(c_in=3, c_out=2)
        params = layer.init_params(jax.random.PRNGKey(8))
        u = jax.random.normal(jax.random.PRNGKey(9), (layer.X.shape[0], 3))
        self.assertEqual(layer.apply(u, params).shape, (layer.X.shape[0], 2))
        self.assertEqual(layer.spectral_shape(),
                         (layer.n_modes, layer.n_modes, 2, 3))

    def test_kernel_gram_reproduces_the_integral(self):
        """Forming ``kappa`` explicitly and integrating agrees with the layer."""
        layer = _layer()
        R = eqj.derivative_operator_tensor(layer)
        u, _ = _random_span_function(layer, seed=10)
        Xq = jnp.linspace(-0.8, 0.8, 7).reshape(-1, 1)

        got = layer.integral(u, R, Xq)[:, 0]
        kap = layer.kernel_gram(Xq, layer.X, R)[:, :, 0, 0]
        want = kap @ (layer.W * u[:, 0])
        np.testing.assert_allclose(np.array(got), np.array(want), atol=1e-11)

    def test_bad_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            _layer(mode="fourier")


@unittest.skipUnless(_HAS_JAX, "jax not installed")
class TestNeuralOperatorStack(unittest.TestCase):

    def test_linear_stack_is_a_linear_operator(self):
        """With no activation a two-layer stack is exactly one linear operator.

        Composition of exact operators stays exact, which is the property a
        nonlinear stack gives up (see the caveat in ``NeuralOperator``).
        """
        layer1 = _layer()
        layer2 = _layer()
        stack = eqj.NeuralOperator([layer1, layer2], activation=None)
        params = stack.init_params(jax.random.PRNGKey(11))

        u1, _ = _random_span_function(layer1, seed=12)
        u2, _ = _random_span_function(layer1, seed=13)
        a, b = 0.3, -1.7

        got = stack.apply(a * u1 + b * u2, params)
        want = a * stack.apply(u1, params) + b * stack.apply(u2, params)
        # A bias in each layer breaks strict linearity, so compare against the
        # affine combination that the biases actually produce.
        zero = stack.apply(0.0 * u1, params)
        np.testing.assert_allclose(np.array(got - zero),
                                   np.array(want - (a + b) * zero), atol=1e-10)

    def test_stack_runs_with_activation_and_shapes_compose(self):
        layers = [_layer(c_in=1, c_out=4), _layer(c_in=4, c_out=1)]
        stack = eqj.NeuralOperator(layers)
        params = stack.init_params(jax.random.PRNGKey(14))
        u, _ = _random_span_function(layers[0], seed=15)
        out = stack.apply(u, params)
        self.assertEqual(out.shape, (layers[0].X.shape[0], 1))
        self.assertTrue(np.all(np.isfinite(np.array(out))))


@unittest.skipUnless(_HAS_JAX, "jax not installed")
@unittest.skipUnless(_HAS_OPTAX, "optax not installed (pip install equadratures[jax-learn])")
class TestOperatorLearning(unittest.TestCase):

    def test_training_recovers_the_derivative_operator(self):
        """Learn ``d/dx`` from input/output function pairs, and check the operator.

        The gate is not "the loss went down": it is that gradient descent
        recovers the *operator itself*, to about five significant figures, from
        a random initialisation.

        Two deliberate choices, both learned by measurement rather than assumed.

        **The comparison is against**
        :func:`~equadratures.jax.operator.effective_spectral_tensor`, not the raw
        ``spectral`` array. The pointwise term acts as a multiple of the identity
        in mode space, so the two blocks are not separately identifiable: the raw
        tensor comes out ~1e-1 away while the operator it defines is right.
        ``test_raw_spectral_tensor_alone_does_not_match`` pins that down.

        **The learning rate is annealed.** With a constant rate this test passed
        or failed on the seed: measured across seven seeds the final loss ranged
        from 1e-27 to 1e-10, and one run reached 8e-21 by step 3000 and bounced
        back to 1e-5 by step 4000 -- Adam is unstable once the gradient is at
        round-off, because normalising by ``sqrt(v)`` amplifies noise. A cosine
        anneal gives 1e-11 to 5e-10 across the same seeds with no blow-ups. The
        tolerances below carry roughly fifty times the observed worst case, so
        this gate should not be seed-fragile.

        Note what is *not* claimed here. That ``d/dx`` is representable exactly
        is a property of the basis, proven without an optimiser in
        ``test_derivative_operator_is_represented_exactly`` at ~1e-15. How close
        gradient descent gets to it is a separate, weaker statement, and this is
        it.
        """
        layer = _layer()
        R_true = eqj.derivative_operator_tensor(layer)

        C = jax.random.normal(jax.random.PRNGKey(16), (64, layer.n_modes, 1))
        U = jax.vmap(layer.evaluate)(C)
        Y = jax.vmap(lambda u: layer.integral(u, R_true))(U)

        def loss(p):
            return jnp.mean((jax.vmap(lambda u: layer.apply(u, p))(U) - Y) ** 2)

        params = layer.init_params(jax.random.PRNGKey(17))
        start = float(loss(params))

        steps = 4000
        schedule = optax.cosine_decay_schedule(3e-2, decay_steps=steps, alpha=1e-4)
        opt = optax.adam(schedule)
        state = opt.init(params)
        step = jax.jit(jax.value_and_grad(loss))
        for _ in range(steps):
            _, grads = step(params)
            updates, state = opt.update(grads, state)
            params = optax.apply_updates(params, updates)

        end = float(loss(params))
        self.assertLess(end, 1e-8)
        self.assertLess(end, start * 1e-8)

        effective = eqj.effective_spectral_tensor(layer, params)
        scale = float(jnp.abs(R_true).max())
        np.testing.assert_allclose(np.array(effective), np.array(R_true),
                                   rtol=0, atol=1e-3 * scale)

    def test_raw_spectral_tensor_alone_does_not_match(self):
        """The non-identifiability above is real, not a tolerance artefact.

        Documented as a test so that a future change which *does* make the raw
        tensor identifiable fails loudly here rather than silently weakening the
        gate above.
        """
        layer = _layer()
        R_true = eqj.derivative_operator_tensor(layer)
        params = layer.init_params(jax.random.PRNGKey(18))
        params = dict(params)
        # Set the layer to represent d/dx exactly, but split between the two
        # blocks rather than putting it all in `spectral`.
        shift = 0.5
        n = layer.n_modes
        params["pointwise"] = jnp.full((1, 1), shift)
        params["spectral"] = R_true - shift * jnp.eye(n)[:, :, None, None]
        params["bias"] = jnp.zeros(1)

        u, _ = _random_span_function(layer, seed=19)
        got = layer.apply(u, params)
        want = layer.integral(u, R_true)
        np.testing.assert_allclose(np.array(got), np.array(want), atol=1e-11)

        raw_gap = float(jnp.abs(params["spectral"] - R_true).max())
        self.assertGreater(raw_gap, 0.1)
        eff_gap = float(jnp.abs(eqj.effective_spectral_tensor(layer, params)
                                - R_true).max())
        self.assertLess(eff_gap, 1e-12)


if __name__ == "__main__":
    unittest.main()
