"""``Parameter`` -- a 1-D random input with a differentiable standard<->physical map.

Wraps a *standard* orthonormal family (uniform on [-1, 1], or standard normal)
together with an affine map to the physical domain. Because the map is a plain
differentiable function of its defining parameters (bounds, or mean/std), moments
computed through the pipeline are differentiable **w.r.t. the distribution
parameters** -- not only w.r.t. the data. That closes the loop on the grant's
"auto-differentiable" thesis (e.g. sensitivity of an expectation to an input's
range).
"""
import functools

import jax.numpy as jnp

from equadratures.jax.recurrence import (
    uniform_recurrence,
    hermite_recurrence,
    legendre_recurrence,
    stieltjes_recurrence,
)
from equadratures.jax.quadrature import gauss_quadrature


@functools.lru_cache(maxsize=None)
def _legendre_grid(n_grid):
    """Gauss--Legendre nodes/weights on ``[-1, 1]`` (cached; they are constants).

    These do not depend on any traced value, so caching them is both safe and
    worth it: the discretisation is otherwise the dominant cost of every
    gradient step through a density.
    """
    alpha, beta, mu0 = legendre_recurrence(n_grid)
    return gauss_quadrature(alpha, beta, mu0)


def density_recurrence(density, lower, upper, n, n_grid=200, normalise=True):
    """Recurrence coefficients of a measure given by a *density* on an interval.

    This is the last link in the differentiable chain. ``stieltjes_recurrence``
    already turns a discretised measure into recurrence coefficients; what was
    missing was a differentiable way to *build* that discretisation from a
    density. With this, a distribution's own parameters (a Beta shape, a
    truncation bound) become inputs you can differentiate a UQ output against --
    not just the data.

    The interval is discretised with **Gauss--Legendre** nodes rather than an
    equispaced grid. Two reasons, both practical:

    * far higher accuracy per node for smooth densities;
    * the nodes are strictly *interior*, so densities that diverge at an
      endpoint -- Beta with a shape parameter below 1, say -- are never
      evaluated at the singularity. The classic equispaced grid in
      ``distributions/recurrence_utils.py`` evaluates the endpoints and returns
      ``inf`` for exactly those cases.

    Parameters
    ----------
    density : callable
        ``x -> density values``, evaluated on an array of points inside
        ``[lower, upper]``. Need not be normalised. Anything it closes over is
        differentiable.
    lower, upper : float
        Interval of support. Differentiable.
    n : int
        Number of recurrence coefficients (static).
    n_grid : int
        Number of discretisation nodes (static). Raise it for rough densities.
    normalise : bool
        Scale to unit mass, giving a probability measure (``mu0 = 1``) so that
        ``Poly`` reads off ``mean = c_0`` directly. Matches what classic does
        internally.

    Returns
    -------
    alpha, beta, mu0
        Ready for :func:`equadratures.jax.quadrature.gauss_quadrature`.
    """
    x_std, w_std = _legendre_grid(n_grid)
    half = 0.5 * (upper - lower)
    x = half * (x_std + 1.0) + lower
    w = w_std * half * density(x)
    if normalise:
        w = w / jnp.sum(w)
    return stieltjes_recurrence(x, w, n)


def beta_density(shape_a, shape_b, lower=0.0, upper=1.0):
    """Beta density on ``[lower, upper]``, differentiable in its shape parameters.

    The normalising constant is omitted: :func:`density_recurrence` normalises
    numerically, so ``B(a, b)`` would cancel anyway.
    """
    def density(x):
        t = (x - lower) / (upper - lower)
        return t ** (shape_a - 1.0) * (1.0 - t) ** (shape_b - 1.0)
    return density


def truncated_gaussian_density(mean, std):
    """Gaussian density, differentiable in ``mean`` and ``std``.

    Truncation comes from the interval passed to :func:`density_recurrence`;
    normalisation there handles the missing tail mass.
    """
    def density(x):
        z = (x - mean) / std
        return jnp.exp(-0.5 * z ** 2)
    return density


class Parameter:
    """A 1-D random parameter.

    Parameters
    ----------
    distribution : {'uniform', 'gaussian'/'normal'}
    lower, upper : float
        Physical support for ``uniform``.
    mean, std : float
        Physical mean/standard deviation for ``gaussian``.
    """

    def __init__(self, distribution="uniform", lower=-1.0, upper=1.0,
                 mean=0.0, std=1.0):
        self.distribution = distribution.lower()
        self.lower = lower
        self.upper = upper
        self.mean = mean
        self.std = std

    def recurrence(self, n):
        """Standard-domain, probability-normalised recurrence ``(alpha, beta, mu0)``."""
        if self.distribution == "uniform":
            return uniform_recurrence(n)
        if self.distribution in ("gaussian", "normal"):
            return hermite_recurrence(n)
        raise ValueError("unknown distribution %r" % self.distribution)

    def to_standard(self, x_physical):
        """Physical -> standard-domain points (differentiable in the bounds/moments)."""
        if self.distribution == "uniform":
            return (2.0 * x_physical - (self.lower + self.upper)) / (self.upper - self.lower)
        return (x_physical - self.mean) / self.std

    def to_physical(self, x_standard):
        """Standard-domain -> physical points (differentiable in the bounds/moments)."""
        if self.distribution == "uniform":
            return 0.5 * (x_standard * (self.upper - self.lower) + (self.lower + self.upper))
        return self.mean + self.std * x_standard


class DensityParameter:
    """A 1-D random parameter defined by an arbitrary density.

    Where :class:`Parameter` covers the two families with analytic recurrences,
    this covers everything else: hand it a density and it builds the orthonormal
    family numerically, differentiably. It exposes the same ``recurrence(n)``
    interface, so it drops straight into
    :class:`equadratures.jax.poly.Poly`.

    Because the density is an ordinary closure, the whole downstream pipeline --
    quadrature, basis, fit, mean, variance, Sobol' indices -- is differentiable
    with respect to whatever parameters that closure captures.

    Parameters
    ----------
    density : callable
        ``x -> density values`` on ``[lower, upper]``. Unnormalised is fine.
    lower, upper : float
        Interval of support.
    n_grid : int
        Discretisation nodes for the Stieltjes procedure (static).

    Examples
    --------
    Sensitivity of a surrogate's variance to a Beta shape parameter::

        def variance_given(a):
            p = DensityParameter(beta_density(a, 3.0), 0.0, 1.0)
            rec = [p.recurrence(6)]
            X, W = tensor_quadrature(rec)
            poly = Poly(rec, total_order_indices(1, 4))
            poly.fit_projection(X, f(X), W)
            return poly.variance()

        jax.grad(variance_given)(2.0)
    """

    def __init__(self, density, lower, upper, n_grid=200):
        self.density = density
        self.lower = lower
        self.upper = upper
        self.n_grid = n_grid

    def measure(self):
        """The discretised measure ``(nodes, weights)`` as a probability measure."""
        x_std, w_std = _legendre_grid(self.n_grid)
        half = 0.5 * (self.upper - self.lower)
        x = half * (x_std + 1.0) + self.lower
        w = w_std * half * self.density(x)
        return x, w / jnp.sum(w)

    def recurrence(self, n):
        """Probability-normalised recurrence ``(alpha, beta, mu0)``."""
        return density_recurrence(self.density, self.lower, self.upper, n,
                                  n_grid=self.n_grid, normalise=True)
