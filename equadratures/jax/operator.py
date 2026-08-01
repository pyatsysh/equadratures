"""Neural-operator layers whose integral operator is a polynomial kernel.

A neural operator learns a map between *functions* rather than between vectors.
Every published architecture is a stack of the same layer,

    v(x) = sigma( W u(x) + (K u)(x) + b ),      (K u)(x) = int kappa(x, y) u(y) dy

and they differ only in how the integral operator ``K`` is represented. The
Fourier neural operator takes ``kappa`` diagonal in a Fourier basis and applies
it with an FFT. This module takes ``kappa`` in the **orthonormal polynomial
basis** of :mod:`equadratures.jax` and applies it by **Gauss quadrature**, which
is what this library is for:

    kappa(x, y) = sum_{j,k} R_{jk} phi_j(x) phi_k(y)

    (K u)(x) = sum_{j,k} R_{jk} phi_j(x) c_k,   c_k = sum_i w_i phi_k(x_i) u(x_i)

The coefficients ``c_k`` are a spectral projection, so for ``u`` in the span the
projection is *exact* rather than approximate, and the whole layer is a finite
sum of differentiable operations. Three consequences are worth stating because
they are the reason to prefer this representation over an FFT:

**It is interpretable.** ``R`` is the operator's matrix in a basis of known
polynomial modes, which is how a spectral method has always represented an
operator. A learned ``R`` can be read: which input modes feed which output
modes, and how strongly.

**It is exact on the span.** Any linear operator mapping the span into itself
has an exact ``R`` (obtainable as ``R_{jk} = <phi_j, A phi_k>``), so the layer
represents it to machine precision rather than approximating it. Differentiation
is one such operator; see the acceptance gate in ``tests/test_jax_operator.py``.

**It is discretisation-invariant, exactly.** The parameters live on polynomial
modes, not on a grid. Evaluating a trained layer on a finer quadrature returns
the *same function*, to round-off, and not merely a similar one.

Two parameterisations of ``R`` are offered:

``mode="diagonal"``
    ``R`` is diagonal in the modes and dense across channels, shape
    ``(n_modes, c_out, c_in)``. This is the direct analogue of the FNO layer.
    In the single-channel case it is exactly the Mercer kernel of
    :class:`~equadratures.jax.kernel.PolynomialKernel` with spectrum
    ``R[k] = theta_k^2``, and so is positive semi-definite when ``R >= 0``.

``mode="dense"``
    ``R`` couples modes as well as channels, shape
    ``(n_modes, n_modes, c_out, c_in)``. This is strictly more expressive: a
    mode-diagonal kernel is symmetric, so it cannot represent a non-symmetric
    operator such as differentiation, and dense mode can.

Functions are represented throughout as arrays of shape ``(n_points,
n_channels)`` sampled at the layer's own quadrature nodes. Pure JAX: everything
here is differentiable, ``jit``-able and ``vmap``-able.
"""
import numpy as np
import jax
import jax.numpy as jnp

from equadratures.jax.basis import design_matrix
from equadratures.jax.poly import tensor_quadrature


class SpectralOperatorLayer:
    """One neural-operator layer over an orthonormal polynomial basis.

    Parameters
    ----------
    recurrences : sequence of ``(alpha, beta, mu0)``, length ``d``
        Per-dimension recurrence coefficients. Their length sets the quadrature
        rule used for the projection, so make it large enough that the rule is
        exact for the products you care about (a rule of ``q`` nodes is exact to
        degree ``2q - 1``).
    indices : array_like, shape (n_modes, d)
        Multi-index set naming the polynomial modes ``phi_k``, e.g. from
        :func:`~equadratures.jax.basis.total_order_indices`.
    channels_in, channels_out : int
        Channel widths. A scalar-valued function is one channel.
    mode : {"dense", "diagonal"}
        Parameterisation of the spectral tensor ``R``; see the module docstring.
    """

    def __init__(self, recurrences, indices, channels_in=1, channels_out=1,
                 mode="dense"):
        if mode not in ("dense", "diagonal"):
            raise ValueError("mode must be 'dense' or 'diagonal', got %r" % (mode,))
        self.recurrences = recurrences
        self.indices = np.asarray(indices)
        self.n_modes = self.indices.shape[0]
        self.channels_in = int(channels_in)
        self.channels_out = int(channels_out)
        self.mode = mode

        # The layer owns its quadrature and its design matrix at the nodes:
        # both are fixed by construction, so they are built once rather than on
        # every call.
        self.X, self.W = tensor_quadrature(recurrences)
        self.Phi = design_matrix(self.X, self.indices, self.recurrences)

    # ------------------------------------------------------------------ setup

    def spectral_shape(self):
        """Shape of the spectral tensor ``R`` for this layer's ``mode``."""
        if self.mode == "diagonal":
            return (self.n_modes, self.channels_out, self.channels_in)
        return (self.n_modes, self.n_modes, self.channels_out, self.channels_in)

    def init_params(self, key, scale=None):
        """Random initial parameters.

        ``scale`` defaults to ``1 / sqrt(n_modes * channels_in)``, the usual
        fan-in scaling, which keeps the spectral term the same order as the
        pointwise term at initialisation instead of swamping it.
        """
        k_spec, k_point = jax.random.split(key)
        fan_in = self.n_modes * self.channels_in
        if scale is None:
            scale = 1.0 / np.sqrt(fan_in)
        return {
            "spectral": scale * jax.random.normal(k_spec, self.spectral_shape()),
            "pointwise": scale * jax.random.normal(
                k_point, (self.channels_out, self.channels_in)),
            "bias": jnp.zeros(self.channels_out),
        }

    # ------------------------------------------------------- spectral transforms

    def project(self, u):
        """Spectral coefficients of ``u`` sampled at the layer's nodes.

        ``c_k = sum_i w_i phi_k(x_i) u(x_i)``, exact for ``u`` in the span when
        the quadrature is exact to twice the basis degree.

        Parameters
        ----------
        u : array_like, shape (n_nodes, channels_in)

        Returns
        -------
        jax.numpy.ndarray, shape (n_modes, channels_in)
        """
        u = jnp.asarray(u)
        return self.Phi.T @ (self.W[:, None] * u)

    def design(self, Xq=None):
        """Design matrix at ``Xq``, or at the layer's own nodes if ``Xq`` is None."""
        if Xq is None:
            return self.Phi
        return design_matrix(Xq, self.indices, self.recurrences)

    def evaluate(self, coeffs, Xq=None):
        """Reconstruct a function from its coefficients at ``Xq``."""
        return self.design(Xq) @ coeffs

    # ------------------------------------------------------------- the operator

    def apply_spectral(self, coeffs, R):
        """Apply the spectral tensor to coefficients: the operator, in mode space."""
        if self.mode == "diagonal":
            return jnp.einsum("koi,ki->ko", R, coeffs)
        return jnp.einsum("jkoi,ki->jo", R, coeffs)

    def integral(self, u, R, Xq=None):
        """The integral operator ``(K u)(Xq)`` alone, without the pointwise term."""
        return self.evaluate(self.apply_spectral(self.project(u), R), Xq)

    def kernel_gram(self, X1, X2, R):
        """The kernel ``kappa(X1, X2)`` itself, for inspection.

        Returned with shape ``(len(X1), len(X2), channels_out, channels_in)``.
        Building it is never necessary -- :meth:`integral` never forms it -- but
        a learned operator is easier to trust when you can look at its kernel.
        """
        P1 = self.design(X1)
        P2 = self.design(X2)
        if self.mode == "diagonal":
            return jnp.einsum("aj,bj,joi->aboi", P1, P2, R)
        return jnp.einsum("aj,bk,jkoi->aboi", P1, P2, R)

    def apply(self, u, params, Xq=None, activation=None):
        """Evaluate the full layer ``sigma(W u + K u + b)``.

        The pointwise term uses the *polynomial representation* of ``u`` rather
        than its samples, so that the layer is a map between functions and can
        be evaluated at query points where ``u`` was never sampled. On the
        layer's own nodes the two agree, exactly, whenever ``u`` is in the span.

        Parameters
        ----------
        u : array_like, shape (n_nodes, channels_in)
        params : dict with keys ``spectral``, ``pointwise``, ``bias``
        Xq : array_like, shape (n_query, d), optional
            Where to evaluate the output. Defaults to the layer's own nodes,
            which is what stacking needs.
        activation : callable, optional
            Applied elementwise to the result. ``None`` leaves the layer linear.

        Returns
        -------
        jax.numpy.ndarray, shape (n_query, channels_out)
        """
        coeffs = self.project(u)
        Pq = self.design(Xq)
        out = Pq @ self.apply_spectral(coeffs, params["spectral"])
        out = out + (Pq @ coeffs) @ params["pointwise"].T
        out = out + params["bias"]
        if activation is not None:
            out = activation(out)
        return out


class NeuralOperator:
    """A stack of :class:`SpectralOperatorLayer` s sharing one quadrature.

    The last layer is left linear, as is standard: a nonlinearity on the output
    would restrict the range for no gain.

    A caveat worth stating rather than burying. Between layers the output is
    re-projected onto the polynomial span, and ``sigma(v)`` is generally *not*
    in the span even when ``v`` is. So a nonlinear stack carries a truncation
    error at every layer, controlled by the mode count. This is not a defect of
    the polynomial basis -- the Fourier neural operator truncates in exactly the
    same way -- but it does mean a purely *linear* stack is exact on the span
    and a nonlinear one is not.

    Parameters
    ----------
    layers : sequence of SpectralOperatorLayer
    activation : callable
        Applied between layers, not after the last one.
    """

    def __init__(self, layers, activation=jax.nn.tanh):
        self.layers = list(layers)
        # "Sharing one quadrature" is a requirement, not a suggestion: each
        # layer projects its input at its *own* nodes, so a stack over
        # different rules hands layer k+1 samples taken at points it will
        # treat as its own. With different node counts that is a shape error;
        # with equal counts but different nodes it is silently wrong output,
        # which is the worse failure. Hence the check.
        first = self.layers[0].X if self.layers else None
        for i, layer in enumerate(self.layers[1:], start=1):
            if (layer.X.shape != first.shape
                    or bool(jnp.any(jnp.abs(layer.X - first) > 1e-12))):
                raise ValueError(
                    "layer %d uses a different quadrature from layer 0; a "
                    "NeuralOperator stack must share one rule, because each "
                    "layer re-projects at its own nodes" % i)
        self.activation = activation

    def init_params(self, key):
        """One parameter dict per layer."""
        keys = jax.random.split(key, len(self.layers))
        return [layer.init_params(k) for layer, k in zip(self.layers, keys)]

    def apply(self, u, params, Xq=None):
        """Push ``u`` through the stack; ``Xq`` applies to the final layer only."""
        v = u
        last = len(self.layers) - 1
        for i, (layer, p) in enumerate(zip(self.layers, params)):
            v = layer.apply(v, p,
                            Xq=Xq if i == last else None,
                            activation=None if i == last else self.activation)
        return v


def effective_spectral_tensor(layer, params):
    """The layer's total linear operator, spectral and pointwise terms combined.

    The pointwise term ``W u(x)`` is *itself* representable in the spectral
    tensor: on the polynomial representation it acts as ``W`` times the identity
    in mode space. So ``spectral`` and ``pointwise`` are not separately
    identifiable, and two trainings that agree perfectly on every function will
    generally disagree on the raw ``spectral`` array.

    This matters in practice. Comparing a learned ``params["spectral"]`` against
    a known operator will report a discrepancy that is not an error; comparing
    the tensor returned here will not. The acceptance gate in
    ``tests/test_jax_operator.py`` does the latter, and recovers ``d/dx`` to
    about 1e-14 where the raw comparison is off by 1e-1.

    Returns
    -------
    jax.numpy.ndarray
        Shaped like a ``mode="dense"`` tensor, ``(n_modes, n_modes, c_out,
        c_in)``, whatever the layer's own mode, since a diagonal tensor plus a
        pointwise term is a dense operator.
    """
    n = layer.n_modes
    R = params["spectral"]
    if layer.mode == "diagonal":
        dense = jnp.einsum("jk,koi->jkoi", jnp.eye(n), R)
    else:
        dense = R
    return dense + jnp.einsum("jk,oi->jkoi", jnp.eye(n), params["pointwise"])


def linear_operator_tensor(layer, operator):
    """The exact spectral tensor of a known linear operator, for ``mode="dense"``.

    Computes ``R_{jk} = <phi_j, A phi_k>`` by the layer's own quadrature, which
    is exact whenever that rule integrates the products ``phi_j * (A phi_k)``
    exactly. Use it to check a learned operator against the truth, or to build a
    layer analytically instead of training one.

    Parameters
    ----------
    layer : SpectralOperatorLayer
        Must be single-channel; the tensor returned is shaped for it.
    operator : callable
        Maps an array of points ``(m, d)`` to the values of ``A phi_k`` at those
        points, with shape ``(m, n_modes)`` -- i.e. it acts on the whole basis at
        once. See :func:`derivative_operator_tensor` for the common case.

    Returns
    -------
    jax.numpy.ndarray, shape (n_modes, n_modes, 1, 1)
    """
    A_phi = operator(layer.X)                       # (m, n_modes)
    R = layer.Phi.T @ (layer.W[:, None] * A_phi)    # (n_modes, n_modes)
    return R[:, :, None, None]


def derivative_operator_tensor(layer, dimension=0):
    """Exact spectral tensor of ``d/dx_dimension``, by differentiating the basis.

    Differentiation maps a polynomial of degree ``p`` to one of degree ``p - 1``,
    so it maps the span into itself and has an exact representation here. The
    basis derivative is taken by autodiff rather than by a recurrence identity,
    which keeps this correct for any measure the library supports rather than
    only for Legendre.
    """
    def basis_derivative(X):
        def row(x):
            return design_matrix(x[None, :], layer.indices, layer.recurrences)[0]
        return jax.vmap(jax.jacfwd(row))(X)[:, :, dimension]   # (m, n_modes)

    return linear_operator_tensor(layer, basis_derivative)
