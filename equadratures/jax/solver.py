"""Differentiable sparse / regularised solves for the coefficient system ``A c = y``.

The classic ``equadratures.solver`` reaches for an external convex-programming
stack (``cvxpy``) for its ``compressed-sensing`` and ``elastic-net`` methods.
That is a dead end for a differentiable pipeline: the solve is an opaque
sub-process, so no gradient flows back to the design matrix, the data, or the
regularisation strength.

Here the same problems are solved natively in JAX and made differentiable by
**implicit differentiation** rather than by unrolling the iterations. The
forward pass is FISTA (accelerated proximal gradient); the backward pass never
looks at the iterates at all -- it differentiates the *optimality conditions* of
the converged solution. That gives gradients which are

* **exact** -- not an approximation that improves with iteration count,
* **cheap** -- one linear solve on the active set, independent of ``max_iter``,
* **memory-flat** -- no tape over thousands of iterations.

The problem solved is the elastic net

.. math::

    \\min_c \\; \\tfrac12 \\| A c - y \\|_2^2
            + \\lambda_1 \\| c \\|_1
            + \\tfrac12 \\lambda_2 \\| c \\|_2^2 ,

which specialises to the LASSO (:math:`\\lambda_2 = 0`, see :func:`lasso`) and to
ridge (:math:`\\lambda_1 = 0`). The LASSO is the penalised form of basis-pursuit
denoising, i.e. the compressed-sensing recovery used for polynomial chaos with
few samples: for every noise budget :math:`\\varepsilon` in the constrained form
there is a :math:`\\lambda_1` giving the same solution.

Why implicit differentiation is exact here
------------------------------------------
Write :math:`S = \\{ i : c_i \\ne 0 \\}` for the active set and
:math:`s = \\operatorname{sign}(c_S)`. The sub-gradient optimality conditions of
the elastic net read

.. math::

    A_S^\\top (A_S c_S - y) + \\lambda_2 c_S + \\lambda_1 s = 0 ,
    \\qquad
    | [A^\\top (Ac - y)]_i | \\le \\lambda_1 \\;\\; (i \\notin S) .

The inactive conditions are strict inequalities at a generic solution, so a small
perturbation of :math:`(A, y, \\lambda_1, \\lambda_2)` leaves both :math:`S` and
:math:`s` unchanged. The active block is then a smooth square system, and the
implicit function theorem applies to it directly. With
:math:`H = A_S^\\top A_S + \\lambda_2 I` and :math:`r = Ac - y`, the
vector-Jacobian products for a cotangent :math:`\\bar c` are, writing
:math:`u = H^{-1} \\bar c_S` (extended by zero off the active set),

.. math::

    \\bar y = A u, \\quad
    \\bar\\lambda_1 = -s^\\top u, \\quad
    \\bar\\lambda_2 = -c^\\top u, \\quad
    \\bar A = -\\left( r u^\\top + (A u) c^\\top \\right) .

The active set enters only through :math:`u`, which vanishes off :math:`S`, so
the masked forms above need no gather/scatter and stay ``jit``-friendly.

Non-differentiable points (a coefficient exactly entering or leaving the active
set) form a measure-zero set, exactly as for ``jnp.abs`` at the origin; JAX's
usual convention of returning a one-sided value applies.
"""
from functools import partial

import jax
import jax.numpy as jnp

# FISTA iterations used by default. Implicit differentiation decouples gradient
# accuracy from this number -- it only has to be large enough to pin down the
# active set and the coefficient values themselves.
DEFAULT_MAX_ITER = 2000


def soft_threshold(v, t):
    """Proximal operator of ``t * ||.||_1``: ``sign(v) * max(|v| - t, 0)``.

    Produces *exact* zeros, which is what makes the active set well defined.
    """
    return jnp.sign(v) * jnp.maximum(jnp.abs(v) - t, 0.0)


def _fista(A, y, lam_l1, lam_l2, max_iter):
    """Accelerated proximal gradient (FISTA) for the elastic net.

    Step size ``1 / L`` with ``L = ||A||_2^2 + lam_l2`` the Lipschitz constant of
    the smooth part's gradient, which guarantees monotone convergence.
    """
    n = A.shape[1]
    L = jnp.linalg.norm(A, ord=2) ** 2 + lam_l2
    eta = 1.0 / L
    thresh = eta * lam_l1

    def body(_, state):
        c, z, t = state
        grad = A.T @ (A @ z - y) + lam_l2 * z
        c_new = soft_threshold(z - eta * grad, thresh)
        t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t ** 2))
        z_new = c_new + ((t - 1.0) / t_new) * (c_new - c)
        return c_new, z_new, t_new

    c0 = jnp.zeros(n, dtype=A.dtype)
    c, _, _ = jax.lax.fori_loop(0, max_iter, body, (c0, c0, jnp.array(1.0, A.dtype)))
    return c


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def elastic_net(A, y, lam_l1, lam_l2, max_iter=DEFAULT_MAX_ITER):
    """Elastic-net solve, differentiable in ``A``, ``y``, ``lam_l1`` and ``lam_l2``.

    Minimises ``0.5 ||A c - y||^2 + lam_l1 ||c||_1 + 0.5 lam_l2 ||c||^2``.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Design matrix (e.g. from :func:`equadratures.jax.basis.design_matrix`).
    y : array_like, shape (m,)
        Observations.
    lam_l1 : float
        ``l1`` penalty -- drives sparsity. ``0`` gives a pure ridge solve.
    lam_l2 : float
        ``l2`` penalty -- stabilises correlated columns. ``0`` gives the LASSO.
    max_iter : int, optional
        FISTA iterations (static). Gradients do not depend on this beyond
        convergence of the solution itself.

    Returns
    -------
    jax.numpy.ndarray, shape (n,)
        Coefficients, with exact zeros off the active set.

    Notes
    -----
    Gradients come from implicit differentiation of the optimality conditions
    (see the module docstring), not from unrolling FISTA.
    """
    return _fista(A, y, lam_l1, lam_l2, max_iter)


def _elastic_net_fwd(A, y, lam_l1, lam_l2, max_iter):
    c = _fista(A, y, lam_l1, lam_l2, max_iter)
    return c, (A, y, lam_l1, lam_l2, c)


def _elastic_net_bwd(max_iter, res, cbar):
    A, y, lam_l1, lam_l2, c = res
    del lam_l1, max_iter                      # enter only via the active set

    # Active set: soft-thresholding produces exact zeros, so this is unambiguous.
    active = c != 0.0
    mask = active.astype(A.dtype)

    # Solve H u = cbar on the active block, u = 0 elsewhere. Padding the inactive
    # block with the identity keeps the system square, non-singular and
    # statically shaped -- the padded rows return the zeros we want.
    n = A.shape[1]
    M = mask[:, None] * mask[None, :]
    H = M * (A.T @ A + lam_l2 * jnp.eye(n, dtype=A.dtype))
    H = H + jnp.diag(1.0 - mask)
    u = jnp.linalg.solve(H, mask * cbar)

    r = A @ c - y
    Au = A @ u

    A_bar = -(jnp.outer(r, u) + jnp.outer(Au, c))
    y_bar = Au
    lam_l1_bar = -jnp.sum(jnp.sign(c) * u)
    lam_l2_bar = -jnp.sum(c * u)
    return A_bar, y_bar, lam_l1_bar, lam_l2_bar


elastic_net.defvjp(_elastic_net_fwd, _elastic_net_bwd)


def lasso(A, y, lam, max_iter=DEFAULT_MAX_ITER):
    """LASSO / basis-pursuit-denoising solve (``l1``-penalised least squares).

    The sparse-recovery workhorse for polynomial chaos from few samples, and the
    differentiable stand-in for the classic ``compressed-sensing`` solver.
    Thin wrapper over :func:`elastic_net` with ``lam_l2 = 0``.
    """
    return elastic_net(A, y, lam, 0.0, max_iter)


def lasso_debiased(A, y, lam, max_iter=DEFAULT_MAX_ITER):
    """LASSO support selection followed by an unpenalised re-fit on that support.

    The ``l1`` penalty that buys sparsity also shrinks the surviving
    coefficients towards zero, biasing them by an amount proportional to
    ``lam``. The standard remedy in compressed sensing is to use the ``l1``
    solve only to *choose* the active set, then re-fit ordinary least squares
    restricted to it. That recovers the coefficients essentially exactly when
    the support is correct.

    Differentiability: the active set is piecewise constant in ``(A, y, lam)``,
    so it contributes no gradient, and the re-fit is a plain linear solve that
    autodiff already handles. The derivative w.r.t. ``lam`` is therefore exactly
    zero -- correct, since ``lam`` acts only through the choice of support.

    Returns
    -------
    jax.numpy.ndarray, shape (n,)
        Coefficients, zero off the selected support.
    """
    A = jnp.asarray(A)
    y = jnp.asarray(y)
    c_l1 = jax.lax.stop_gradient(lasso(A, y, lam, max_iter))
    mask = (c_l1 != 0.0).astype(A.dtype)

    n = A.shape[1]
    M = mask[:, None] * mask[None, :]
    H = M * (A.T @ A) + jnp.diag(1.0 - mask)
    return jnp.linalg.solve(H, mask * (A.T @ y))


def ridge(A, y, lam):
    """Ridge / Tikhonov solve ``(A^T A + lam I) c = A^T y``.

    Smooth and dense, so plain autodiff through the linear solve is already
    exact -- no implicit-differentiation machinery needed.
    """
    A = jnp.asarray(A)
    n = A.shape[1]
    return jnp.linalg.solve(A.T @ A + lam * jnp.eye(n, dtype=A.dtype), A.T @ y)


def lasso_path(A, y, lams, max_iter=DEFAULT_MAX_ITER):
    """Solve the LASSO at several penalties at once via ``vmap``.

    Returns an array of shape ``(len(lams), n)``. Useful for sparsity sweeps and
    for picking a penalty by cross-validation -- and, being a ``vmap``, it runs
    the whole path in one batched pass.
    """
    return jax.vmap(lambda lam: lasso(A, y, lam, max_iter))(jnp.asarray(lams))
