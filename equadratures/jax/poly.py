"""A minimal differentiable polynomial approximation (``Poly``) over the
``equadratures.jax`` primitives.

Mirrors the essentials of the classic ``equadratures.Poly`` workflow -- build a
multivariate orthonormal basis, fit coefficients to data, predict, and read off
mean/variance -- but as a pure JAX object: fitting, prediction and the moments
are all differentiable (w.r.t. inputs, data and, through the recurrences,
distribution parameters), ``jit``-able and ``vmap``-able.

Moments assume *probability* measures (``mu0 = 1``, e.g.
:func:`~equadratures.jax.recurrence.uniform_recurrence` /
:func:`~equadratures.jax.recurrence.hermite_recurrence`), so ``p_0 = 1`` and, for
an orthonormal basis, ``mean = c_0`` and ``variance = sum_{k>0} c_k^2``.
"""
import numpy as np
import jax.numpy as jnp

from equadratures.jax.basis import design_matrix
from equadratures.jax.quadrature import gauss_quadrature
from equadratures.jax.solver import (
    DEFAULT_MAX_ITER,
    elastic_net,
    lasso,
    lasso_debiased,
    ridge,
)


def tensor_quadrature(recurrences):
    """Tensor-product Gauss quadrature from per-dimension recurrence coefficients.

    Parameters
    ----------
    recurrences : sequence of ``(alpha, beta, mu0)``
        One per dimension, each already sized for the desired 1-D rule.

    Returns
    -------
    X : jax.numpy.ndarray, shape (prod q_i, d)
    W : jax.numpy.ndarray, shape (prod q_i,)
        Nodes and (product) weights; ``W`` sums to ``prod mu0_i``.
    """
    nodes_1d, weights_1d = [], []
    for alpha, beta, mu0 in recurrences:
        nd, wt = gauss_quadrature(alpha, beta, mu0)
        nodes_1d.append(nd)
        weights_1d.append(wt)
    node_grids = jnp.meshgrid(*nodes_1d, indexing="ij")
    weight_grids = jnp.meshgrid(*weights_1d, indexing="ij")
    X = jnp.stack([g.ravel() for g in node_grids], axis=1)
    W = jnp.ones(X.shape[0])
    for wg in weight_grids:
        W = W * wg.ravel()
    return X, W


class Poly:
    """Differentiable orthonormal polynomial model.

    Parameters
    ----------
    recurrences : sequence of ``(alpha, beta, mu0)``, length ``d``
        Per-dimension recurrence coefficients (sized ``>= max index + 1``).
    indices : array_like, shape (n_basis, d)
        Multi-index set, e.g. from
        :func:`equadratures.jax.basis.total_order_indices`.
    """

    def __init__(self, recurrences, indices):
        self.recurrences = recurrences
        self.indices = np.asarray(indices)
        self.coefficients = None

    def get_design(self, X):
        """Design matrix at ``X`` (shape ``(m, n_basis)``)."""
        return design_matrix(X, self.indices, self.recurrences)

    def fit(self, X, y):
        """Least-squares regression fit (any sample points)."""
        A = self.get_design(X)
        self.coefficients = jnp.linalg.lstsq(A, jnp.asarray(y), rcond=None)[0]
        return self.coefficients

    def fit_ridge(self, X, y, regularisation):
        """Tikhonov / ridge-regularised least squares (differentiable).

        Solves ``(AᵀA + reg·I) c = Aᵀy``. Useful when the design is
        ill-conditioned or under-determined; ``regularisation`` is itself a
        differentiable input (e.g. a learnable hyper-parameter).
        """
        A = self.get_design(X)
        self.coefficients = ridge(A, jnp.asarray(y), regularisation)
        return self.coefficients

    def fit_lasso(self, X, y, regularisation, max_iter=DEFAULT_MAX_ITER):
        """Sparse ``l1``-regularised fit -- the compressed-sensing route.

        Recovers a sparse expansion from fewer samples than basis terms. Unlike
        the classic ``compressed-sensing`` solver (an opaque ``cvxpy`` call),
        this one is differentiable in the data, the design and the penalty, via
        implicit differentiation of the optimality conditions. See
        :mod:`equadratures.jax.solver`.
        """
        A = self.get_design(X)
        self.coefficients = lasso(A, jnp.asarray(y), regularisation, max_iter)
        return self.coefficients

    def fit_lasso_debiased(self, X, y, regularisation, max_iter=DEFAULT_MAX_ITER):
        """Sparse fit with the ``l1`` shrinkage bias removed.

        Uses the ``l1`` solve to pick the active basis terms, then re-fits them
        by ordinary least squares. This is usually what you want for a
        compressed-sensing surrogate: the sparsity of the ``l1`` solution with
        the accuracy of an unpenalised fit.
        """
        A = self.get_design(X)
        self.coefficients = lasso_debiased(A, jnp.asarray(y), regularisation,
                                           max_iter)
        return self.coefficients

    def fit_elastic_net(self, X, y, l1, l2, max_iter=DEFAULT_MAX_ITER):
        """Elastic-net fit: ``l1`` for sparsity, ``l2`` for correlated columns.

        Differentiable in both penalties, so they can be *learned* (e.g. tuned
        by gradient descent on a validation loss) rather than grid-searched.
        """
        A = self.get_design(X)
        self.coefficients = elastic_net(A, jnp.asarray(y), l1, l2, max_iter)
        return self.coefficients

    def fit_projection(self, X, y, weights):
        """Spectral projection ``c_k = sum_q w_q y_q p_k(x_q)`` -- exact for
        functions in the span when ``(X, weights)`` is a quadrature exact to the
        required degree."""
        A = self.get_design(X)
        self.coefficients = A.T @ (jnp.asarray(weights) * jnp.asarray(y))
        return self.coefficients

    def predict(self, X):
        """Evaluate the fitted model at ``X``."""
        return self.get_design(X) @ self.coefficients

    def mean(self):
        """Mean of the expansion (``c_0``; probability measure assumed)."""
        return self.coefficients[0]

    def variance(self):
        """Variance of the expansion (``sum_{k>0} c_k^2``; Parseval)."""
        return jnp.sum(self.coefficients[1:] ** 2)

    def sobol_indices(self):
        """First-order Sobol' indices, one per dimension.

        ``S_i`` is the fraction of the variance explained by basis terms that
        depend on dimension ``i`` *alone*. Differentiable w.r.t. the coefficients
        (hence w.r.t. the training data) -- a differentiable UQ output.
        """
        c2 = self.coefficients ** 2
        var = jnp.sum(c2[1:])
        total_deg = self.indices.sum(axis=1)
        S = []
        for i in range(self.indices.shape[1]):
            only_i = jnp.asarray((self.indices[:, i] > 0) & (total_deg == self.indices[:, i]))
            S.append(jnp.sum(jnp.where(only_i, c2, 0.0)) / var)
        return jnp.stack(S)

    def total_sobol_indices(self):
        """Total-effect Sobol' indices: variance fraction from *all* terms that
        involve dimension ``i`` (main effect plus every interaction)."""
        c2 = self.coefficients ** 2
        var = jnp.sum(c2[1:])
        T = []
        for i in range(self.indices.shape[1]):
            has_i = jnp.asarray(self.indices[:, i] > 0)
            T.append(jnp.sum(jnp.where(has_i, c2, 0.0)) / var)
        return jnp.stack(T)
