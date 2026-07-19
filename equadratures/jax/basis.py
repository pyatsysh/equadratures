"""Multi-index sets and the multivariate orthonormal design matrix.

Multi-index generation is combinatorial and static (plain NumPy, not traced).
The design matrix is a differentiable, ``vmap``/``jit``-able JAX function of the
sample points: each basis column is the product over dimensions of the 1-D
orthonormal polynomials from :mod:`equadratures.jax.polynomial`.
"""
import itertools

import numpy as np
import jax.numpy as jnp

from equadratures.jax.polynomial import orthonormal_polynomials


def total_order_indices(dimensions, order):
    """Total-order multi-index set: all ``d``-tuples with ``sum <= order``.

    Returned sorted by total degree then lexicographically, so the constant
    term ``(0, ..., 0)`` is row 0. Shape ``(n_basis, dimensions)``, dtype int.
    """
    idx = [t for t in itertools.product(range(order + 1), repeat=dimensions)
           if sum(t) <= order]
    idx.sort(key=lambda t: (sum(t),) + t)
    return np.asarray(idx, dtype=int)


def tensor_grid_indices(dimensions, order):
    """Tensor-product multi-index set: all ``d``-tuples with each entry ``<= order``."""
    idx = list(itertools.product(range(order + 1), repeat=dimensions))
    idx.sort(key=lambda t: (sum(t),) + t)
    return np.asarray(idx, dtype=int)


def design_matrix(X, indices, recurrences):
    """Multivariate orthonormal-polynomial design (Vandermonde) matrix.

    Parameters
    ----------
    X : array_like, shape (m, d)
        Sample points.
    indices : array_like, shape (n_basis, d)
        Multi-index set (e.g. from :func:`total_order_indices`).
    recurrences : sequence of ``(alpha, beta, mu0)``, length ``d``
        Per-dimension recurrence coefficients.

    Returns
    -------
    jax.numpy.ndarray, shape (m, n_basis)
        Column ``j`` is ``prod_dim p_{indices[j, dim]}(X[:, dim])``.
        Differentiable w.r.t. ``X``.
    """
    X = jnp.asarray(X)
    idx = np.asarray(indices)
    m, d = X.shape
    max_deg = int(idx.max()) if idx.size else 0
    terms = jnp.ones((idx.shape[0], m))
    for dim in range(d):
        alpha, beta, mu0 = recurrences[dim]
        P = orthonormal_polynomials(X[:, dim], alpha, beta, mu0, max_deg)  # (max_deg+1, m)
        terms = terms * P[idx[:, dim], :]                                 # (n_basis, m)
    return terms.T                                                        # (m, n_basis)
