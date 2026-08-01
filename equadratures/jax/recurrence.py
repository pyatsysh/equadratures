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


def uniform_recurrence(n):
    """Orthonormal recurrence for the *uniform probability* measure on [-1, 1]
    (density 1/2, ``mu0 = 1``).

    Same ``alpha``/``beta`` as :func:`legendre_recurrence` (the recurrence
    coefficients are independent of the measure's normalisation); only ``mu0``
    differs. Use this in the ``Poly`` layer so that ``p_0 = 1`` and the expansion
    coefficients give ``mean = c_0``, ``variance = sum_{k>0} c_k^2``.
    """
    alpha, beta, _ = legendre_recurrence(n)
    return alpha, beta, 1.0


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


def stieltjes_recurrence(nodes, weights, n):
    """Recurrence coefficients of a *discretised* measure via the Stieltjes
    procedure (JAX port of ``distributions.recurrence_utils`` — differentiable).

    Given a measure represented by ``(nodes, weights)`` (e.g. a fine grid
    weighted by a density), returns the first ``n`` orthonormal-polynomial
    recurrence coefficients. This is what lets an *arbitrary* distribution feed
    the differentiable pipeline: the coefficients are differentiable w.r.t. the
    nodes and weights, hence w.r.t. any distribution parameters behind them.

    Parameters
    ----------
    nodes : array_like, shape (m,)
        Support points of the discretised measure.
    weights : array_like, shape (m,)
        Non-negative masses at ``nodes``. ``mu0`` is their sum.
    n : int
        Number of recurrence coefficients (static).

    Returns
    -------
    alpha : jax.numpy.ndarray, shape (n,)
    beta : jax.numpy.ndarray, shape (n-1,)
    mu0 : jax.numpy.ndarray, scalar

    Raises
    ------
    ValueError
        If ``n`` exceeds the number of support points. A measure supported on
        ``m`` points admits at most ``m`` orthonormal polynomials -- the next one
        would have to vanish at every one of them -- and past that the recurrence
        breaks down. It does not break down *loudly*: the exhausted ``beta``
        comes out at zero and the ones after it come back non-zero, from
        round-off, and the resulting quadrature rule has positive weights summing
        correctly to ``mu0``. Measured with 4 nodes and 8 requested terms:
        ``beta = [0.745, 0.596, 0.447, 0.0, 0.956, 0.195, 0.341]``, of which the
        last three are noise wearing a plausible costume. Hence the check.

    Notes
    -----
    Accurate to order ``n`` only if the discretisation integrates polynomials up
    to degree ``2n-1`` well enough (e.g. an ``M>=n`` point Gauss rule makes the
    first ``n`` coefficients exact).

    The check above counts support *points*, not distinct ones. A measure with
    repeated nodes, or with some weights at zero, supports fewer polynomials
    than it has entries, and that case is not detected here.
    """
    x = jnp.asarray(nodes)
    w = jnp.asarray(weights)
    if int(n) > x.shape[0]:
        raise ValueError(
            "cannot build %d recurrence coefficients from a measure supported "
            "on %d points: at most %d orthonormal polynomials exist, and past "
            "that the recurrence returns round-off that looks like data"
            % (int(n), x.shape[0], x.shape[0]))
    mu0 = jnp.sum(w)

    alpha = [jnp.sum(w * x) / mu0]        # alpha_0 = weighted mean
    b = [mu0]                             # b_0 = mu0 (0th moment)
    p_prev = jnp.zeros_like(x)            # p_{-1}
    p_cur = jnp.ones_like(x)             # p_0 (monic)
    s = mu0
    for k in range(1, n):
        p_next = (x - alpha[k - 1]) * p_cur - b[k - 1] * p_prev
        s1 = jnp.sum(w * p_next ** 2)
        alpha.append(jnp.sum(w * x * p_next ** 2) / s1)
        b.append(s1 / s)
        s = s1
        p_prev, p_cur = p_cur, p_next

    b = jnp.stack(b)
    return jnp.stack(alpha), jnp.sqrt(b[1:]), mu0
