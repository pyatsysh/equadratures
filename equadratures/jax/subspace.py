"""Latent-variable models: learning the subspace a model actually varies on.

Many expensive models with dozens of inputs vary along only a handful of
directions. Writing ``U`` for a ``d x r`` matrix with orthonormal columns, the
assumption is

    f(x)  ~=  g(U^T x)

with ``r`` far smaller than ``d``. The columns of ``U`` are the **latent
variables**: not a subset of the inputs but linear combinations of them, learned
from data. This is the third of the three things the grant proposal asks the JAX
backend to enable -- "learnable kernels, learnable expansions, latent-variable
models" -- and it is the one where differentiability changes the method rather
than merely speeding it up.

Two routes are provided, and they answer different questions.

:func:`active_subspace` uses gradients. The matrix ``C = E[grad f grad f^T]``
has small eigenvalues in directions the model ignores, so its dominant
eigenvectors are the active ones. Classically the gradients are the obstacle:
you either have an adjoint solver or you pay ``d`` finite differences per
sample. Here they are one ``jax.grad`` of the surrogate, exactly, which is the
whole argument of this project applied to dimension reduction.

:class:`PolynomialRidge` learns ``U`` and ``g`` together from function values
alone, with no gradients of the expensive model required. It uses **variable
projection**: for any ``U`` the best polynomial coefficients are a linear least
squares, so they can be eliminated in closed form and the loss becomes a
function of ``U`` alone. The classic namespace does this too
(``Subspaces(method='variable-projection')``) with a hand-derived Gauss-Newton
Jacobian, an Armijo line search and two hand-tuned constants whose own source
comment asks "How do we know these are the best values of gamma and beta?". Here
the derivative comes from autodiff through the least-squares solve, so there is
no Jacobian to derive and no line-search constants to guess.

**On identifiability.** ``U`` is only defined up to rotation within its own
column space: ``U`` and ``UQ`` for orthogonal ``Q`` describe the same subspace
and give the identical model. So never compare ``U`` matrices elementwise --
compare subspaces with :func:`subspace_distance`, which is invariant to that
rotation. This is the same trap as the spectral/pointwise split in
:mod:`equadratures.jax.operator`, in a different costume.
"""
import numpy as np
import jax
import jax.numpy as jnp

from equadratures.jax.basis import design_matrix, total_order_indices
from equadratures.jax.recurrence import hermite_recurrence


def orthonormalise(A):
    """Map an unconstrained ``d x r`` matrix to one with orthonormal columns.

    A thin QR, sign-fixed so the map is continuous: ``numpy``/LAPACK leave the
    sign of each column of ``Q`` free, and letting it flip mid-optimisation puts
    a discontinuity in the middle of the loss.

    Raises if asked for more columns than rows. A thin QR would quietly return
    ``min(d, r)`` columns instead, which is not an error anywhere downstream --
    it just produces a model on fewer latent variables than the caller asked
    for, silently.

    Optimising over this parameterisation rather than on the Stiefel manifold
    directly means the constraint is satisfied by construction and any ordinary
    optimiser will do -- no manifold machinery, no retraction step. The classic
    namespace reaches for ``pymanopt`` for this.
    """
    if A.ndim != 2:
        raise ValueError("expected a 2-D matrix, got shape %r" % (A.shape,))
    if A.shape[1] > A.shape[0]:
        raise ValueError(
            "cannot orthonormalise %d columns in %d dimensions: a subspace "
            "cannot have more directions than the space containing it"
            % (A.shape[1], A.shape[0]))
    Q, R = jnp.linalg.qr(A)
    return Q * jnp.sign(jnp.diag(R))


def subspace_distance(U, V):
    """Distance between the column spaces of ``U`` and ``V``, in [0, 1].

    ``||U U^T - V V^T||_2``. Zero when the two span the same subspace, whatever
    rotation or sign convention each happens to use, which is why this and not
    an elementwise comparison is the right way to check a learned subspace.
    """
    U = jnp.asarray(U)
    V = jnp.asarray(V)
    return jnp.linalg.norm(U @ U.T - V @ V.T, 2)


def gradient_covariance(f, X, weights=None):
    """``C = E[grad f grad f^T]`` at the sample points, by autodiff.

    Parameters
    ----------
    f : callable
        Maps ``(m, d)`` points to ``(m,)`` values. Must be JAX-traceable; a
        fitted :class:`~equadratures.jax.poly.Poly` prediction qualifies.
    X : array_like, shape (m, d)
    weights : array_like, shape (m,), optional
        Quadrature weights. Uniform if omitted.

    Returns
    -------
    jax.numpy.ndarray, shape (d, d)
    """
    X = jnp.asarray(X)
    m = X.shape[0]
    w = jnp.full((m,), 1.0 / m) if weights is None else jnp.asarray(weights)
    scalar = lambda x: f(x[None, :])[0]
    grads = jax.vmap(jax.grad(scalar))(X)                   # (m, d)
    return jnp.einsum("i,ij,ik->jk", w, grads, grads)


def active_subspace(f, X, weights=None, dimension=None):
    """Active subspace of ``f``: the directions it actually varies along.

    Returns
    -------
    eigenvalues : jax.numpy.ndarray, shape (d,)
        Descending. A gap after entry ``r`` is the evidence that ``r`` latent
        variables suffice; no gap means this model does not have a low-
        dimensional structure and you should not pretend otherwise.
    U : jax.numpy.ndarray, shape (d, dimension or d)
        Leading eigenvectors, orthonormal.
    """
    C = gradient_covariance(f, X, weights)
    eigenvalues, vectors = jnp.linalg.eigh(C)
    order = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    vectors = vectors[:, order]
    if dimension is not None:
        vectors = vectors[:, :dimension]
    return eigenvalues, vectors


class PolynomialRidge:
    """``f(x) ~= g(U^T x)`` with ``U`` and the polynomial ``g`` learned together.

    Parameters
    ----------
    dimensions : int
        Input dimension ``d``.
    subspace_dimension : int
        Latent dimension ``r``, usually 1 or 2.
    order : int
        Total-degree order of the polynomial on the latent space.
    recurrence : tuple, optional
        Recurrence for the latent coordinates. Defaults to probabilists'
        Hermite, which is the natural choice: the latent coordinates are
        standardised to zero mean and unit variance below, and for Gaussian
        inputs an orthonormal projection of them is standard normal exactly.
    regularisation : float, optional
        Relative Tikhonov floor on the inner least-squares solve. **Default 0**,
        which uses an exact `lstsq` and is what you want for a well-posed
        problem: on a genuine ridge function it recovers the subspace to about
        1e-16, and any floor above zero costs digits.

        Raise it only if the design is rank-deficient -- fewer distinct sample
        points than basis terms, duplicated rows, or a projection that collapses
        the latent coordinate. There, `lstsq` produces a **NaN gradient** (the
        SVD derivative is undefined with repeated singular values), so the fit
        silently poisons on the next step. ``1e-10`` restores finite gradients
        and costs roughly ten digits of residual, which on an ill-posed problem
        you did not have anyway. This is deliberately not the default: paying
        that everywhere to protect against a pathological input would be the
        wrong trade.

    Notes
    -----
    The latent coordinates are standardised by their own **mean and standard
    deviation** rather than mapped from their min and max. Both put the
    coordinates where the basis expects them, but min/max is not differentiable
    at the sample attaining it and puts a kink in the loss; mean/sd is smooth
    everywhere. The classic implementation uses min/max.
    """

    def __init__(self, dimensions, subspace_dimension, order, recurrence=None,
                 regularisation=0.0):
        if int(subspace_dimension) > int(dimensions):
            raise ValueError(
                "subspace_dimension (%d) cannot exceed dimensions (%d)"
                % (int(subspace_dimension), int(dimensions)))
        if int(subspace_dimension) < 1:
            raise ValueError("subspace_dimension must be at least 1")
        if float(regularisation) < 0.0:
            raise ValueError("regularisation must be non-negative")
        self.dimensions = int(dimensions)
        self.subspace_dimension = int(subspace_dimension)
        self.regularisation = float(regularisation)
        self.order = int(order)
        self.indices = total_order_indices(self.subspace_dimension, self.order)
        if recurrence is None:
            recurrence = hermite_recurrence(self.order + 1)
        self.recurrences = [recurrence] * self.subspace_dimension
        self.U = None
        self.coefficients = None

    # ------------------------------------------------------------- internals

    _VARIANCE_FLOOR = 1e-12

    @classmethod
    def _standardise(cls, Z):
        """Centre and scale, with a gradient that survives zero variance.

        The floor goes **inside** the square root. Writing ``std(z) + eps``
        looks equivalent and is not: ``d sqrt(v)/dv`` is unbounded at ``v = 0``,
        so a latent coordinate with no spread -- constant data, or a projection
        that collapses -- produces a NaN gradient and poisons the whole fit on
        the very next step. ``sqrt(v + eps)`` is finite everywhere.
        """
        centred = Z - Z.mean(axis=0)
        scale = jnp.sqrt(jnp.mean(centred ** 2, axis=0) + cls._VARIANCE_FLOOR)
        return centred / scale

    def _latent_design(self, A, X):
        """Design matrix on the standardised latent coordinates."""
        U = orthonormalise(A)
        Z = jnp.asarray(X) @ U                              # (m, r)
        return design_matrix(self._standardise(Z), self.indices,
                             self.recurrences), U

    def _solve(self, V, y):
        """Inner least squares, exact by default and floored when asked."""
        if self.regularisation == 0.0:
            return jnp.linalg.lstsq(V, y, rcond=None)[0]
        n = V.shape[1]
        gram = V.T @ V
        floor = self.regularisation * jnp.trace(gram) / n
        return jnp.linalg.solve(gram + floor * jnp.eye(n, dtype=V.dtype), V.T @ y)

    def loss(self, A, X, y):
        """Variable-projection residual: coefficients eliminated in closed form.

        For a given ``U`` the optimal coefficients are ``V^+ y``, so the loss is
        a function of ``U`` alone. Differentiating it means differentiating
        *through* a least-squares solve, which autodiff does and a hand-derived
        Gauss-Newton Jacobian is written to avoid.
        """
        V, _ = self._latent_design(A, X)
        y = jnp.asarray(y)
        coefficients = self._solve(V, y)
        return jnp.mean((V @ coefficients - y) ** 2)

    # ------------------------------------------------------------------- API

    def fit(self, X, y, steps=600, learning_rate=5e-2, key=None, A0=None,
            optimiser=None):
        """Learn the subspace by gradient descent, then the coefficients exactly.

        ``optimiser`` takes an Optax optimiser; without one this falls back to
        plain gradient descent so that the base ``[jax]`` extra is enough. Optax
        is a better bet on real problems and the anneal matters -- see the
        neural-operator module for what a constant rate does near a minimum.
        """
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        if A0 is None:
            key = jax.random.PRNGKey(0) if key is None else key
            A = jax.random.normal(key, (self.dimensions, self.subspace_dimension))
        else:
            A = jnp.asarray(A0)

        objective = jax.jit(jax.value_and_grad(lambda a: self.loss(a, X, y)))
        history = []

        if optimiser is None:
            for _ in range(steps):
                value, grad = objective(A)
                history.append(float(value))
                A = A - learning_rate * grad
        else:
            import optax
            state = optimiser.init(A)
            for _ in range(steps):
                value, grad = objective(A)
                history.append(float(value))
                updates, state = optimiser.update(grad, state)
                A = optax.apply_updates(A, updates)

        self.U = orthonormalise(A)
        V, _ = self._latent_design(A, X)
        self.coefficients = self._solve(V, y)
        self._A = A
        self._fit_X = X
        return np.asarray(history)

    def predict(self, X):
        """Evaluate the fitted ridge model at ``X``.

        The latent standardisation uses the *training* projections, so that
        prediction is a fixed function rather than one that shifts with whatever
        batch it is handed.
        """
        if self.coefficients is None:
            raise RuntimeError("call fit() before predict()")
        Z_train = self._fit_X @ self.U
        centred = Z_train - Z_train.mean(axis=0)
        scale = jnp.sqrt(jnp.mean(centred ** 2, axis=0) + self._VARIANCE_FLOOR)
        Z = (jnp.asarray(X) @ self.U - Z_train.mean(axis=0)) / scale
        return design_matrix(Z, self.indices, self.recurrences) @ self.coefficients
