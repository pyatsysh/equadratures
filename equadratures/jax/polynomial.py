"""Differentiable evaluation of orthonormal polynomials.

Uses the same symmetric three-term recurrence as the Jacobi matrix, so the
orthonormal polynomials ``p_0, ..., p_degree`` are consistent with the Gauss
quadrature in :mod:`equadratures.jax.quadrature`::

    p_0(x)      = 1 / sqrt(mu0)
    p_{k+1}(x)  = [ (x - alpha_k) p_k(x) - beta_k p_{k-1}(x) ] / beta_{k+1}

The evaluation is a pure function of the points and the recurrence coefficients,
hence differentiable w.r.t. both, ``jit``-able and ``vmap``-able. ``degree`` is a
static Python int (it sets the length of the unrolled recurrence).
"""
import jax.numpy as jnp


def orthonormal_polynomials(x, alpha, beta, mu0, degree):
    """Evaluate orthonormal polynomials up to ``degree`` at points ``x``.

    Parameters
    ----------
    x : array_like, shape (m,)
        Evaluation points.
    alpha : array_like, shape (>= degree,)
        Diagonal recurrence coefficients (``alpha_0, alpha_1, ...``).
    beta : array_like, shape (>= degree,)
        Off-diagonal recurrence coefficients (``beta_1, beta_2, ...``).
    mu0 : float
        Zeroth moment (total mass) of the measure.
    degree : int
        Highest polynomial degree to evaluate.

    Returns
    -------
    jax.numpy.ndarray, shape (degree + 1, m)
        Row ``k`` is ``p_k`` evaluated at ``x``.
    """
    x = jnp.asarray(x)
    rows = [jnp.ones_like(x) / jnp.sqrt(mu0)]                      # p_0
    if degree >= 1:
        rows.append((x - alpha[0]) * rows[0] / beta[0])           # p_1
    for k in range(1, degree):
        rows.append(((x - alpha[k]) * rows[k] - beta[k - 1] * rows[k - 1])
                    / beta[k])                                    # p_{k+1}
    return jnp.stack(rows, axis=0)
