"""Differentiable Gauss quadrature via the Golub--Welsch algorithm.

Given the symmetric three-term recurrence of a family of orthonormal
polynomials, the ``n``-point Gauss nodes are the eigenvalues of the ``n x n``
symmetric tridiagonal Jacobi matrix, and the weights are ``mu0`` times the
squared first components of its (orthonormal) eigenvectors.

``jax.numpy.linalg.eigh`` carries autodiff rules, so nodes and weights are
differentiable with respect to the recurrence coefficients -- and hence with
respect to any distribution parameters those coefficients depend on. This is
the primitive that makes the whole pipeline learnable.
"""
import jax.numpy as jnp


def jacobi_matrix(alpha, beta):
    """Symmetric tridiagonal Jacobi matrix from recurrence coefficients.

    Parameters
    ----------
    alpha : array_like, shape (n,)
        Diagonal recurrence coefficients.
    beta : array_like, shape (n-1,)
        Off-diagonal recurrence coefficients (non-negative).

    Returns
    -------
    jax.numpy.ndarray, shape (n, n)
    """
    return jnp.diag(alpha) + jnp.diag(beta, 1) + jnp.diag(beta, -1)


def gauss_quadrature(alpha, beta, mu0):
    """``n``-point Gauss quadrature nodes and weights (Golub--Welsch).

    Parameters
    ----------
    alpha : array_like, shape (n,)
        Diagonal recurrence coefficients.
    beta : array_like, shape (n-1,)
        Off-diagonal recurrence coefficients.
    mu0 : float
        Zeroth moment (total mass) of the measure, ``integral d mu``.

    Returns
    -------
    nodes : jax.numpy.ndarray, shape (n,)
        Quadrature nodes, ascending.
    weights : jax.numpy.ndarray, shape (n,)
        Quadrature weights (sum to ``mu0``).

    Notes
    -----
    Differentiable w.r.t. ``alpha``, ``beta`` and ``mu0``. An ``n``-point rule
    integrates polynomials up to degree ``2n - 1`` exactly against the measure.
    """
    J = jacobi_matrix(alpha, beta)
    eigvals, eigvecs = jnp.linalg.eigh(J)
    nodes = eigvals
    weights = mu0 * eigvecs[0, :] ** 2
    return nodes, weights


def _monic_at(alpha, beta, x, n):
    """Monic orthogonal polynomials ``pi_{n-2}(x)``, ``pi_{n-1}(x)`` at a point.

    Runs the monic three-term recurrence
    ``pi_{k+1} = (x - alpha_k) pi_k - beta_k pi_{k-1}`` with
    ``beta_k = beta[k-1]**2`` (our ``beta`` holds the *symmetric* Jacobi
    off-diagonals, i.e. the square roots of the monic recurrence's).

    Used to place a prescribed node: both the Radau and the Lobatto
    construction need the last two monic polynomials evaluated at the node.
    """
    p_prev = jnp.zeros_like(jnp.asarray(x))          # pi_{-1}
    p_cur = jnp.ones_like(jnp.asarray(x))            # pi_0
    for k in range(n - 1):
        b = beta[k - 1] ** 2 if k >= 1 else 0.0      # pi_{-1} = 0, so beta_0 is free
        p_prev, p_cur = p_cur, (x - alpha[k]) * p_cur - b * p_prev
    return p_prev, p_cur


def radau_quadrature(alpha, beta, mu0, end):
    """``n``-point Gauss--Radau rule with one prescribed node at ``end``.

    A Radau rule fixes one node -- normally an endpoint of the support -- and
    chooses the remaining ``n - 1`` optimally. That costs one degree of
    exactness relative to Gauss (``2n - 2`` instead of ``2n - 1``) but is what
    you want when the integrand must be sampled *at* the boundary, e.g. an
    initial condition or a design constraint that lives on the edge of the
    domain.

    Following Golub (1973): the rule is Gauss quadrature for a Jacobi matrix
    whose final diagonal entry has been retuned so that ``end`` is forced into
    the spectrum. Only ``alpha[-1]`` changes, so the construction -- and hence
    the whole rule -- stays differentiable in the recurrence coefficients and in
    ``end`` itself.

    Parameters
    ----------
    alpha : array_like, shape (n,)
    beta : array_like, shape (n-1,)
    mu0 : float
        Zeroth moment of the measure.
    end : float
        The node to prescribe.

    Returns
    -------
    nodes, weights : jax.numpy.ndarray, shape (n,)
        ``end`` appears among ``nodes``; ``weights`` sum to ``mu0``.
    """
    alpha = jnp.asarray(alpha)
    beta = jnp.asarray(beta)
    n = alpha.shape[0]
    p_prev, p_cur = _monic_at(alpha, beta, end, n)
    alpha_last = end - beta[n - 2] ** 2 * p_prev / p_cur
    return gauss_quadrature(alpha.at[n - 1].set(alpha_last), beta, mu0)


def lobatto_quadrature(alpha, beta, mu0, lower, upper):
    """``n``-point Gauss--Lobatto rule with both ``lower`` and ``upper`` fixed.

    Fixing both endpoints costs two degrees of exactness (``2n - 3``), and buys
    a rule whose nodes span the closed domain -- the usual choice for spectral
    element methods and for any surrogate that must interpolate at the
    boundaries.

    Both the last diagonal *and* the last off-diagonal of the Jacobi matrix are
    retuned (Golub 1973), which is what pins down two eigenvalues at once. The
    construction is differentiable in the recurrence coefficients and in both
    endpoints.

    Parameters
    ----------
    alpha : array_like, shape (n,)
    beta : array_like, shape (n-1,)
    mu0 : float
    lower, upper : float
        The two nodes to prescribe.

    Returns
    -------
    nodes, weights : jax.numpy.ndarray, shape (n,)
        ``lower`` and ``upper`` are the first and last node.
    """
    alpha = jnp.asarray(alpha)
    beta = jnp.asarray(beta)
    n = alpha.shape[0]
    pl_prev, pl_cur = _monic_at(alpha, beta, lower, n)
    pr_prev, pr_cur = _monic_at(alpha, beta, upper, n)

    det = pl_cur * pr_prev - pr_cur * pl_prev
    alpha_last = (lower * pl_cur * pr_prev - upper * pr_cur * pl_prev) / det
    beta_last_sq = (upper - lower) * pl_cur * pr_cur / det

    alpha_mod = alpha.at[n - 1].set(alpha_last)
    beta_mod = beta.at[n - 2].set(jnp.sqrt(beta_last_sq))
    return gauss_quadrature(alpha_mod, beta_mod, mu0)
