"""Three-term recurrence coefficients for orthonormal polynomial families.

Each factory returns ``(alpha, beta, mu0)`` ready for
:func:`equadratures.jax.quadrature.gauss_quadrature`:

* ``alpha`` -- shape ``(n,)`` diagonal recurrence coefficients,
* ``beta``  -- shape ``(n-1,)`` off-diagonal coefficients,
* ``mu0``   -- zeroth moment (total mass) of the measure.

The classical (analytic) coefficients live here; a general
data-/weight-driven recurrence (the JAX port of the Stieltjes procedure in
``distributions/recurrence_utils.py``) is the next step and will slot in with
the same signature.
"""
import jax.numpy as jnp


def legendre_recurrence(n):
    """Orthonormal Legendre recurrence for the uniform weight ``w(x) = 1`` on
    ``[-1, 1]``.

    ``alpha_k = 0``; ``beta_k = k / sqrt(4 k^2 - 1)``; ``mu0 = 2``.
    The resulting rule is Gauss--Legendre on ``[-1, 1]``.
    """
    k = jnp.arange(1, n, dtype=jnp.float64)
    alpha = jnp.zeros(n, dtype=jnp.float64)
    beta = k / jnp.sqrt(4.0 * k**2 - 1.0)
    mu0 = 2.0
    return alpha, beta, mu0


def hermite_recurrence(n):
    """Orthonormal (probabilists') Hermite recurrence for the standard-normal
    weight ``w(x) = exp(-x^2 / 2) / sqrt(2 pi)`` on the real line.

    ``alpha_k = 0``; ``beta_k = sqrt(k)``; ``mu0 = 1``.
    The resulting rule computes ``E[f(X)]`` for ``X ~ N(0, 1)``.
    """
    k = jnp.arange(1, n, dtype=jnp.float64)
    alpha = jnp.zeros(n, dtype=jnp.float64)
    beta = jnp.sqrt(k)
    mu0 = 1.0
    return alpha, beta, mu0
