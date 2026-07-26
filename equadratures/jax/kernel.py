"""Learnable polynomial (Mercer) kernels and Gaussian-process regression.

The "random polynomial kernel" of the NumFOCUS grant: from an orthonormal
polynomial basis ``phi_k`` build

    k(x, x') = sum_k theta_k^2 phi_k(x) phi_k(x') = Phi(x) diag(theta^2) Phi(x')^T

which is positive semi-definite by construction and differentiable in the
log-spectrum ``log_theta`` (so ``theta_k^2 = exp(2 log_theta_k) > 0``). The
spectral weights sit on *known, interpretable* polynomial modes -- exactly the
"interpretable kernels grounded in polynomial approximation theory" the grant
argues for. Training ``log_theta`` (and a noise level) by gradient descent turns
it into a learnable kernel for GP / kernel regression.

Pure JAX; no optimiser dependency here (train with Optax or plain ``jax.grad``).
"""
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

from equadratures.jax.basis import design_matrix


class PolynomialKernel:
    """Mercer kernel from an orthonormal polynomial feature map.

    Parameters
    ----------
    recurrences : sequence of ``(alpha, beta, mu0)``, length ``d``
    indices : array_like, shape (n_features, d)
        Multi-index set defining the polynomial features ``phi_k``.
    """

    def __init__(self, recurrences, indices):
        self.recurrences = recurrences
        self.indices = np.asarray(indices)
        self.n_features = self.indices.shape[0]

    def features(self, X):
        """Feature map ``Phi(X)`` of shape ``(m, n_features)`` (differentiable)."""
        return design_matrix(X, self.indices, self.recurrences)

    def gram(self, X1, X2, log_theta):
        """Kernel matrix ``k(X1, X2)`` with log-spectrum ``log_theta``."""
        P1 = self.features(X1)
        P2 = self.features(X2)
        w = jnp.exp(2.0 * log_theta)
        return (P1 * w) @ P2.T

    def default_log_theta(self):
        """Zero log-spectrum (``theta_k = 1``) -- a sensible init."""
        return jnp.zeros(self.n_features)


def gp_nlml(kernel, X, y, log_theta, log_noise):
    """Negative log marginal likelihood of a GP with ``kernel`` (Cholesky-based)."""
    m = X.shape[0]
    K = kernel.gram(X, X, log_theta) + jnp.exp(log_noise) * jnp.eye(m)
    L = jnp.linalg.cholesky(K)
    a = jsl.solve_triangular(L, y, lower=True)
    quad = jnp.dot(a, a)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return 0.5 * quad + 0.5 * logdet + 0.5 * m * jnp.log(2.0 * jnp.pi)


def gp_predict(kernel, Xtrain, ytrain, Xtest, log_theta, log_noise):
    """Posterior-mean GP prediction at ``Xtest`` (differentiable)."""
    m = Xtrain.shape[0]
    K = kernel.gram(Xtrain, Xtrain, log_theta) + jnp.exp(log_noise) * jnp.eye(m)
    Ks = kernel.gram(Xtest, Xtrain, log_theta)
    alpha = jnp.linalg.solve(K, ytrain)
    return Ks @ alpha


def gp_predict_with_variance(kernel, Xtrain, ytrain, Xtest, log_theta,
                             log_noise, include_noise=False):
    """Posterior mean *and* variance at ``Xtest`` (both differentiable).

    A Gaussian process whose predictions carry no uncertainty is only doing
    kernel ridge regression; the variance is the point. For a Mercer kernel
    built from a finite feature map the posterior variance is the usual

    ``var(x*) = k(x*, x*) - k(x*, X) [K + sigma^2 I]^-1 k(X, x*)``

    evaluated row by row, so it costs one solve shared across all test points.

    Parameters
    ----------
    include_noise : bool
        If ``True`` return the *predictive* variance for a new noisy
        observation (adds ``sigma^2``); if ``False`` return the variance of the
        latent function, which is what you want for error bars on the model
        itself.

    Returns
    -------
    mean : jax.numpy.ndarray, shape (n_test,)
    var : jax.numpy.ndarray, shape (n_test,)
        Clipped at zero: the algebra is exact but round-off can push a
        near-zero variance very slightly negative.

    Notes
    -----
    With a rank-``n_features`` kernel the variance falls to zero once the
    training data pin down every feature -- correct behaviour for a
    finite-dimensional model, and worth knowing before reading the error bars:
    they describe uncertainty *within the polynomial span*, not model-form error
    outside it.
    """
    m = Xtrain.shape[0]
    noise = jnp.exp(log_noise)
    K = kernel.gram(Xtrain, Xtrain, log_theta) + noise * jnp.eye(m)
    Ks = kernel.gram(Xtest, Xtrain, log_theta)              # (n_test, m)

    L = jnp.linalg.cholesky(K)
    alpha = jsl.solve_triangular(L.T, jsl.solve_triangular(L, ytrain, lower=True),
                                 lower=False)
    mean = Ks @ alpha

    v = jsl.solve_triangular(L, Ks.T, lower=True)           # (m, n_test)
    Phi = kernel.features(Xtest)
    w = jnp.exp(2.0 * log_theta)
    k_diag = jnp.sum(Phi * w * Phi, axis=1)                 # k(x*, x*) per point
    var = k_diag - jnp.sum(v ** 2, axis=0)
    if include_noise:
        var = var + noise
    return mean, jnp.maximum(var, 0.0)
