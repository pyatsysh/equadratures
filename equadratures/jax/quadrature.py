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
