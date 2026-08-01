"""Neural operators whose integral operator is a polynomial kernel.

A neural operator learns a map between functions. This one represents the
integral operator in the orthonormal polynomial basis and applies it by Gauss
quadrature, instead of making it diagonal in a Fourier basis and applying it
with an FFT.

Three demonstrations, in increasing order of how much they need training:

  1. no training at all -- differentiation has an exact representation in this
     basis, so the operator can be *built* rather than fitted;
  2. discretisation invariance -- the same parameters on a finer quadrature
     return the same function, exactly, because they live on modes not on a grid;
  3. learning an operator from data, and recovering the operator itself rather
     than merely a small loss.

Run:  python examples/jax_neural_operator.py
Requires: pip install equadratures[jax-learn]     (optax, for demo 3 only)
"""
import numpy as np
import jax
import jax.numpy as jnp

import equadratures.jax as eqj

ORDER = 5


def build_layer(extra_nodes=3, mode="dense"):
    """A 1-D layer on [-1, 1] with a quadrature comfortably exact for the products."""
    recurrences = [eqj.uniform_recurrence(ORDER + extra_nodes)]
    indices = eqj.total_order_indices(1, ORDER)
    return eqj.SpectralOperatorLayer(recurrences, indices, mode=mode)


def demo_exact_derivative():
    """Build d/dx analytically and check it against autodiff of the same function."""
    print("\n=== 1. an operator that needs no training ===")
    layer = build_layer()
    R = eqj.derivative_operator_tensor(layer)

    coeffs = jax.random.normal(jax.random.PRNGKey(1), (layer.n_modes, 1))
    u = layer.evaluate(coeffs)                       # a function in the span

    got = layer.integral(u, R)[:, 0]
    scalar = lambda x: (layer.design(x[None, :]) @ coeffs)[0, 0]
    want = jax.vmap(jax.grad(scalar))(layer.X)[:, 0]

    scale = float(jnp.abs(want).max())
    err = float(jnp.abs(got - want).max())
    print(f"spectral tensor shape        : {tuple(R.shape)}")
    print(f"max |K u  -  du/dx|          : {err:.3e}")
    print(f"relative to the scale of du/dx: {err / scale:.3e}")
    print("Differentiation maps the span into itself, so it is represented")
    print("exactly. Note it is also NOT symmetric, which is why a mode-diagonal")
    print("kernel could not do this and mode='dense' exists.")


def demo_discretisation_invariance():
    """The same parameters, two different quadratures, one function."""
    print("\n=== 2. discretisation invariance, exactly ===")
    coarse = build_layer(extra_nodes=3)
    fine = build_layer(extra_nodes=9)
    R = eqj.derivative_operator_tensor(coarse)

    coeffs = jax.random.normal(jax.random.PRNGKey(2), (coarse.n_modes, 1))
    Xq = jnp.linspace(-0.9, 0.9, 13).reshape(-1, 1)

    out_coarse = coarse.integral(coarse.evaluate(coeffs), R, Xq)
    out_fine = fine.integral(fine.evaluate(coeffs), R, Xq)
    gap = float(jnp.abs(out_coarse - out_fine).max())

    print(f"coarse rule: {coarse.X.shape[0]:2d} nodes")
    print(f"fine rule  : {fine.X.shape[0]:2d} nodes")
    print(f"max difference at 13 query points: {gap:.3e}")
    print("Not 'similar on a finer grid' -- the same function to round-off,")
    print("because the parameters live on polynomial modes and not on a grid.")


def demo_learning_an_operator():
    """Learn d/dx from function pairs, then check the operator, not the loss."""
    print("\n=== 3. learning an operator from data ===")
    try:
        import optax
    except ImportError:
        print("optax not installed; skipping "
              "(pip install equadratures[jax-learn])")
        return

    layer = build_layer()
    R_true = eqj.derivative_operator_tensor(layer)

    coeffs = jax.random.normal(jax.random.PRNGKey(3), (64, layer.n_modes, 1))
    U = jax.vmap(layer.evaluate)(coeffs)
    Y = jax.vmap(lambda u: layer.integral(u, R_true))(U)

    def loss(params):
        pred = jax.vmap(lambda u: layer.apply(u, params))(U)
        return jnp.mean((pred - Y) ** 2)

    params = layer.init_params(jax.random.PRNGKey(4))
    # The learning rate is annealed rather than held constant. Adam is unstable
    # once the gradient reaches round-off -- dividing by sqrt(v) amplifies noise
    # -- and with a constant rate this demo reached 8e-21 by step 3000 and then
    # bounced back to 1e-5 by step 4000. Annealing costs a few digits of final
    # loss and buys a result that does not depend on the seed.
    steps = 4000
    schedule = optax.cosine_decay_schedule(3e-2, decay_steps=steps, alpha=1e-4)
    opt = optax.adam(schedule)
    state = opt.init(params)
    step = jax.jit(jax.value_and_grad(loss))

    print(f"{'step':>6} {'loss':>12}")
    for it in range(steps + 1):
        value, grads = step(params)
        if it % 1000 == 0:
            print(f"{it:6d} {float(value):12.3e}")
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)

    raw_gap = float(jnp.abs(params["spectral"] - R_true).max())
    effective = eqj.effective_spectral_tensor(layer, params)
    eff_gap = float(jnp.abs(effective - R_true).max())

    print(f"\nfinal loss                         : {float(loss(params)):.3e}")
    print(f"|raw spectral       - R_true|_max  : {raw_gap:.3e}   <-- looks wrong")
    print(f"|effective operator - R_true|_max  : {eff_gap:.3e}   <-- is right")
    print("The pointwise term W acts as a multiple of the identity in mode")
    print("space, so `spectral` and `pointwise` are not separately")
    print("identifiable. The operator they jointly define is what to compare;")
    print("the raw array is not.")
    print("Note the separation of claims: that d/dx is representable EXACTLY is")
    print("a property of the basis, shown in demo 1 without any optimiser. How")
    print("close gradient descent gets to it is a weaker, separate statement,")
    print("and this is that statement.")


def demo_stack():
    """A two-layer stack, and the honest caveat about nonlinearity."""
    print("\n=== 4. stacking layers ===")
    layers = [build_layer(), build_layer()]
    stack = eqj.NeuralOperator(layers)
    params = stack.init_params(jax.random.PRNGKey(5))

    coeffs = jax.random.normal(jax.random.PRNGKey(6), (layers[0].n_modes, 1))
    u = layers[0].evaluate(coeffs)
    out = stack.apply(u, params)
    print(f"input  {tuple(u.shape)}  ->  output {tuple(out.shape)}")

    linear = eqj.NeuralOperator(layers, activation=None)
    a, b = 0.3, -1.7
    u2 = layers[0].evaluate(jax.random.normal(jax.random.PRNGKey(7),
                                              (layers[0].n_modes, 1)))
    zero = linear.apply(0.0 * u, params)
    lhs = linear.apply(a * u + b * u2, params) - zero
    rhs = (a * (linear.apply(u, params) - zero)
           + b * (linear.apply(u2, params) - zero))
    print(f"linear stack, superposition error : {float(jnp.abs(lhs - rhs).max()):.3e}")
    print("A linear stack composes exactly. A nonlinear one does not: sigma(v)")
    print("leaves the polynomial span, so each layer re-projects and truncates.")
    print("That is a property of spectral neural operators generally, not of")
    print("this basis -- the Fourier version truncates in the same place.")


if __name__ == "__main__":
    demo_exact_derivative()
    demo_discretisation_invariance()
    demo_learning_an_operator()
    demo_stack()
