"""Validation for multi-device sharding -- the third leg of deliverable D2.

`jit` and `vmap` are covered by the benchmarks; this covers splitting a batch
across devices. The point of these tests is a distinction worth keeping sharp:

* **Correctness is testable anywhere.** JAX will expose several devices on a
  plain CPU if asked, so "sharding across n devices returns exactly what one
  device returns" can be checked on any machine.
* **Speed-up is not.** Nothing here measures or asserts performance, because the
  development machine has no accelerator, and a timing claim from eight
  simulated CPU devices would be worthless.

To exercise the sharding properly, run with several devices::

    XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        python -m unittest discover -s tests -p "test_jax_parallel.py"

With one device the tests still pass and simply verify the no-op path, which is
itself worth knowing: code written against this module runs unchanged on a
laptop. ``test_actually_sharded_across_devices`` skips itself when only one
device is present, so a single-device run cannot silently look like a
multi-device one.
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


def _fitted_poly():
    recurrences = [eqj.uniform_recurrence(4)] * 2
    indices = eqj.total_order_indices(2, 2)
    X, W = eqj.tensor_quadrature(recurrences)
    poly = eqj.Poly(recurrences, indices)
    f = lambda Z: 1.0 + 2.0 * Z[:, 0] + 3.0 * Z[:, 0] * Z[:, 1]
    poly.fit_projection(X, f(X), W)
    return poly


@unittest.skipUnless(_HAS_JAX, "jax not installed")
class TestBatchPadding(unittest.TestCase):

    def test_divisible_batch_is_untouched(self):
        n_devices = eqj.device_count()
        X = jnp.arange(4 * n_devices, dtype=float).reshape(-1, 1)
        padded, original = eqj.parallel.pad_to_devices(X)
        self.assertEqual(original, X.shape[0])
        self.assertEqual(padded.shape, X.shape)

    def test_indivisible_batch_is_padded_to_a_multiple(self):
        n_devices = eqj.device_count()
        n = 4 * n_devices + 1
        X = jnp.arange(n, dtype=float).reshape(-1, 1)
        padded, original = eqj.parallel.pad_to_devices(X)
        self.assertEqual(original, n)
        self.assertEqual(padded.shape[0] % n_devices, 0)
        self.assertGreaterEqual(padded.shape[0], n)
        np.testing.assert_allclose(np.array(padded[:n]), np.array(X))

    def test_batch_smaller_than_device_count(self):
        """The awkward case: fewer items than devices."""
        n_devices = eqj.device_count()
        if n_devices == 1:
            self.skipTest("needs more than one device to be meaningful")
        X = jnp.arange(n_devices - 1, dtype=float).reshape(-1, 1)
        padded, original = eqj.parallel.pad_to_devices(X)
        self.assertEqual(padded.shape[0] % n_devices, 0)
        self.assertEqual(original, n_devices - 1)

    def test_trim_reverses_pad(self):
        X = jnp.arange(13, dtype=float).reshape(-1, 1)
        padded, original = eqj.parallel.pad_to_devices(X)
        np.testing.assert_allclose(
            np.array(eqj.parallel.trim_from_devices(padded, original)),
            np.array(X))


@unittest.skipUnless(_HAS_JAX, "jax not installed")
class TestShardedResultsMatchSingleDevice(unittest.TestCase):
    """The identity that matters: sharding must not change the answer."""

    def test_sharded_predict_matches(self):
        poly = _fitted_poly()
        rng = np.random.default_rng(0)
        # Divisible, indivisible, and smaller than the device count.
        for n in (8 * eqj.device_count(), 100, 7):
            X = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n, 2)))
            single = poly.predict(X)
            sharded = eqj.sharded_predict(poly, X)
            self.assertEqual(sharded.shape, single.shape)
            np.testing.assert_allclose(np.array(sharded), np.array(single),
                                       atol=1e-13, rtol=0)

    def test_sharded_operator_apply_matches(self):
        recurrences = [eqj.uniform_recurrence(9)]
        indices = eqj.total_order_indices(1, 5)
        layer = eqj.SpectralOperatorLayer(recurrences, indices)
        params = layer.init_params(jax.random.PRNGKey(0))

        coeffs = jax.random.normal(jax.random.PRNGKey(1), (37, layer.n_modes, 1))
        U = jax.vmap(layer.evaluate)(coeffs)

        single = jax.vmap(lambda u: layer.apply(u, params))(U)
        sharded = eqj.sharded_apply(layer, params, U)
        np.testing.assert_allclose(np.array(sharded), np.array(single),
                                   atol=1e-13, rtol=0)

    def test_shard_batched_on_an_arbitrary_function(self):
        fn = lambda x: jnp.sum(jnp.sin(x) ** 2) * jnp.exp(-x[0])
        X = jnp.asarray(np.random.default_rng(2).normal(size=(53, 4)))
        single = jax.vmap(fn)(X)
        sharded = eqj.shard_batched(fn)(X)
        np.testing.assert_allclose(np.array(sharded), np.array(single),
                                   atol=1e-13, rtol=0)

    def test_padding_rows_cannot_leak_into_the_result(self):
        """A padded batch must give the same answer as an unpadded one.

        The padding repeats the final row, so if it ever leaked the last entries
        would be wrong. Comparing an indivisible batch against a single-device
        run of the same batch is what catches that.
        """
        poly = _fitted_poly()
        rng = np.random.default_rng(3)
        n = 8 * eqj.device_count() + 3
        X = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n, 2)))
        sharded = eqj.sharded_predict(poly, X)
        self.assertEqual(sharded.shape[0], n)
        np.testing.assert_allclose(np.array(sharded), np.array(poly.predict(X)),
                                   atol=1e-13, rtol=0)


@unittest.skipUnless(_HAS_JAX, "jax not installed")
class TestActuallySharded(unittest.TestCase):

    def test_actually_sharded_across_devices(self):
        """Confirm the array really is split, not merely computed correctly.

        Without this, a single-device run would pass every test above and give
        no signal that the sharding path had never been taken. Skips itself
        rather than passing vacuously when only one device is present.
        """
        if eqj.device_count() < 2:
            self.skipTest("needs XLA_FLAGS=--xla_force_host_platform_device_count=N")

        poly = _fitted_poly()
        n = 8 * eqj.device_count()
        X = jnp.asarray(np.random.default_rng(4).uniform(-1.0, 1.0, size=(n, 2)))
        result = eqj.sharded_predict(poly, X)

        addressable = result.addressable_shards
        self.assertEqual(len(addressable), eqj.device_count())
        self.assertEqual(sum(s.data.shape[0] for s in addressable), n)


if __name__ == "__main__":
    unittest.main()
