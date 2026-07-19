"""``Parameter`` -- a 1-D random input with a differentiable standard<->physical map.

Wraps a *standard* orthonormal family (uniform on [-1, 1], or standard normal)
together with an affine map to the physical domain. Because the map is a plain
differentiable function of its defining parameters (bounds, or mean/std), moments
computed through the pipeline are differentiable **w.r.t. the distribution
parameters** -- not only w.r.t. the data. That closes the loop on the grant's
"auto-differentiable" thesis (e.g. sensitivity of an expectation to an input's
range).
"""
from equadratures.jax.recurrence import uniform_recurrence, hermite_recurrence


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
