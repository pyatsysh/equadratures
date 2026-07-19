"""equadratures.jax — a JAX-native, auto-differentiable backend for equadratures.

This is the *parallel namespace* introduced by the NumFOCUS grant "Auto-
Differentiable Equadratures". The classic NumPy ``equadratures`` API is left
untouched; everything under ``equadratures.jax`` is pure/functional JAX so that
quadrature, orthogonal-polynomial evaluation and (eventually) polynomial fitting
are differentiable, ``jit``-able and ``vmap``/``pmap``-able.

Install the optional dependency::

    pip install equadratures[jax]

float64 is enabled on import: Gauss quadrature exactness needs double precision,
and JAX defaults to float32.
"""
from jax import config as _config
_config.update("jax_enable_x64", True)

from equadratures.jax.quadrature import jacobi_matrix, gauss_quadrature
from equadratures.jax.recurrence import (
    legendre_recurrence,
    uniform_recurrence,
    hermite_recurrence,
    stieltjes_recurrence,
)
from equadratures.jax.polynomial import orthonormal_polynomials
from equadratures.jax.basis import (
    total_order_indices,
    tensor_grid_indices,
    design_matrix,
)
from equadratures.jax.poly import Poly, tensor_quadrature
from equadratures.jax.parameter import Parameter

__all__ = [
    "jacobi_matrix",
    "gauss_quadrature",
    "legendre_recurrence",
    "uniform_recurrence",
    "hermite_recurrence",
    "stieltjes_recurrence",
    "orthonormal_polynomials",
    "total_order_indices",
    "tensor_grid_indices",
    "design_matrix",
    "Poly",
    "tensor_quadrature",
    "Parameter",
]
