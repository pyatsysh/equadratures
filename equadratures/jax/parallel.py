"""Sharding work across devices, and an honest account of what that buys here.

Deliverable D2 of the grant asks for "parallel, GPU and JIT-compiled
capabilities". `jit` and `vmap` are exercised and benchmarked elsewhere in this
namespace. This module covers the third leg: splitting a batch across **several
devices**.

**What is verified and what is not.** JAX can be told to expose several devices
on a single CPU, so the *correctness* of a sharded computation -- that splitting
a batch across `n` devices returns exactly what one device returns -- is
testable on any machine, and `tests/test_jax_parallel.py` tests it on eight
simulated devices. What is *not* testable without accelerators is the
**speed-up**, and no claim about it is made anywhere in this namespace. Those are
different claims and conflating them would be the easiest way to overstate this
deliverable.

Run the tests with several devices like this::

    XLA_FLAGS=--xla_force_host_platform_device_count=8 python -m unittest ...

With one device everything here still works and simply does not shard, so code
written against it is portable to a machine that does have accelerators.

**Why this is a thin module.** Almost everything in `equadratures.jax` is already
a pure function of a batch, so sharding is a question of *where the array lives*
rather than of rewriting the maths. The useful thing to provide is therefore the
batch bookkeeping -- padding to a whole number of devices, and trimming
afterwards -- which is the part that is fiddly and easy to get wrong.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec


AXIS = "batch"


def device_count():
    """Number of devices JAX can see."""
    return jax.device_count()


def make_mesh(axis_name=AXIS, devices=None):
    """A one-dimensional mesh over the available devices."""
    devices = jax.devices() if devices is None else list(devices)
    return Mesh(np.asarray(devices), (axis_name,))


def pad_to_devices(X, n_devices=None):
    """Pad a leading batch axis up to a whole multiple of the device count.

    Returns ``(padded, original_length)``. Sharding requires the batch to divide
    evenly, and real batch sizes do not oblige; padding with repeats of the last
    row keeps every device doing well-defined work, and
    :func:`trim_from_devices` discards the surplus afterwards. The padding rows
    are never observable in the result.
    """
    X = jnp.asarray(X)
    n = X.shape[0]
    d = device_count() if n_devices is None else int(n_devices)
    remainder = n % d
    if remainder == 0:
        return X, n
    pad = d - remainder
    filler = jnp.repeat(X[-1:], pad, axis=0)
    return jnp.concatenate([X, filler], axis=0), n


def trim_from_devices(Y, original_length):
    """Discard the rows :func:`pad_to_devices` added."""
    return Y[:original_length]


def shard_batched(fn, mesh=None, axis_name=AXIS):
    """Wrap a per-example function so a batch is split across devices.

    Parameters
    ----------
    fn : callable
        Maps one example to one result. It is ``vmap``-ed internally, so write
        it for a single item and let this handle the batch.
    mesh : jax.sharding.Mesh, optional

    Returns
    -------
    callable
        Takes a batched array, returns a batched result. Padding and trimming
        are handled, so the batch length need not divide the device count.

    Notes
    -----
    With a single device this is ``jit(vmap(fn))`` and nothing else, which is
    why code written against it runs unchanged on a laptop and on a multi-GPU
    host.
    """
    mesh = make_mesh(axis_name) if mesh is None else mesh
    sharding = NamedSharding(mesh, PartitionSpec(axis_name))
    batched = jax.jit(jax.vmap(fn),
                      in_shardings=sharding, out_shardings=sharding)

    def run(X):
        padded, n = pad_to_devices(X, mesh.shape[axis_name])
        return trim_from_devices(batched(padded), n)

    return run


def sharded_predict(poly, X, mesh=None):
    """Evaluate a fitted :class:`~equadratures.jax.poly.Poly` across devices.

    The common large-batch operation in this library: a surrogate exists in
    order to be evaluated a great many times, which is exactly the shape of
    problem worth sharding.
    """
    return shard_batched(lambda x: poly.predict(x[None, :])[0], mesh)(X)


def sharded_apply(layer, params, U, mesh=None):
    """Apply a :class:`~equadratures.jax.operator.SpectralOperatorLayer` to a
    batch of input functions, split across devices.

    Operator learning is the natural multi-device workload here: the parameters
    are small and shared while the batch of functions is large.
    """
    return shard_batched(lambda u: layer.apply(u, params), mesh)(U)
