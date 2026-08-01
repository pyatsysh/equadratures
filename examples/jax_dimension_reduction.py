"""Latent-variable models: finding the subspace a model actually varies on.

Many expensive models with dozens of inputs vary along only a handful of
directions. Writing ``U`` for a ``d x r`` matrix with orthonormal columns,

    f(x)  ~=  g(U^T x)

and the columns of ``U`` are the latent variables -- learned linear combinations
of the inputs, not a subset of them.

Four demonstrations:

  1. the gradient route, when you can differentiate the model;
  2. the values-only route, when you cannot -- variable projection, with the
     derivative coming from autodiff through a least-squares solve;
  3. why you must compare *subspaces* and never ``U`` itself;
  4. the negative control -- a model with no low-dimensional structure, which
     must be reported as having none.

Run:  python examples/jax_dimension_reduction.py
Requires: pip install equadratures[jax]
"""
import numpy as np
import jax
import jax.numpy as jnp

import equadratures.jax as eqj

DIMENSIONS = 8


def ridge_problem(seed=0, m=300):
    """A model that genuinely has one latent variable, so there is a truth."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(DIMENSIONS, 1))
    u /= np.linalg.norm(u)
    X = jnp.asarray(rng.normal(size=(m, DIMENSIONS)))
    g = lambda z: 1.0 + 2.0 * z + 0.5 * z ** 2 - 0.3 * z ** 3
    return jnp.asarray(u), X, g(X @ jnp.asarray(u))[:, 0], g


def demo_gradient_route():
    """C = E[grad f grad f^T]: eigenvalues rank the directions."""
    print("\n=== 1. with gradients: the active subspace ===")
    u, X, _, g = ridge_problem()
    f = lambda Z: g(Z @ u)[:, 0]

    eigenvalues, U = eqj.active_subspace(f, X, dimension=1)
    ev = np.array(eigenvalues)
    print("eigenvalues of the gradient covariance:")
    print("  " + "  ".join("%.3e" % v for v in ev[:4]) + "  ...")
    print(f"gap between first and second        : {ev[0] / max(ev[1], 1e-300):.2e}")
    print(f"distance to the true subspace       : "
          f"{float(eqj.subspace_distance(U, u)):.3e}")
    print("One non-zero eigenvalue, because the model has exactly one latent")
    print("variable. The gradients cost one jax.grad, not d finite differences")
    print("per sample, which is what makes this cheap enough to bother with.")


def demo_values_only_route():
    """Variable projection: learn U from function values, no model gradients."""
    print("\n=== 2. without gradients: polynomial ridge ===")
    u, X, y, g = ridge_problem()

    model = eqj.PolynomialRidge(dimensions=DIMENSIONS, subspace_dimension=1,
                                order=3)
    history = model.fit(X, y, steps=800, learning_rate=0.2)

    print(f"loss {history[0]:.3e} -> {history[-1]:.3e} over {len(history)} steps")
    print(f"distance to the true subspace : "
          f"{float(eqj.subspace_distance(model.U, u)):.3e}")

    relative = float(jnp.linalg.norm(model.predict(X) - y) / jnp.linalg.norm(y))
    print(f"training relative error       : {relative:.3e}")

    rng = np.random.default_rng(99)
    X_new = jnp.asarray(rng.normal(size=(150, DIMENSIONS)))
    y_new = g(X_new @ u)[:, 0]
    held = float(jnp.linalg.norm(model.predict(X_new) - y_new)
                 / jnp.linalg.norm(y_new))
    print(f"held-out relative error       : {held:.3e}")
    print("No gradients of the model were used. For any U the best polynomial")
    print("coefficients are a least squares, so they are eliminated in closed")
    print("form and the loss depends on U alone -- autodiff then runs straight")
    print("through the solve. The classic namespace does this with a")
    print("hand-derived Gauss-Newton Jacobian and two tuned constants.")


def demo_identifiability():
    """U is defined only up to rotation. Compare subspaces, not matrices."""
    print("\n=== 3. compare subspaces, never U itself ===")
    U = eqj.orthonormalise(jax.random.normal(jax.random.PRNGKey(0), (6, 2)))
    Q = eqj.orthonormalise(jax.random.normal(jax.random.PRNGKey(1), (2, 2)))
    rotated = U @ Q

    print(f"max |U - UQ| elementwise      : {float(jnp.abs(U - rotated).max()):.3f}"
          "   <- looks completely different")
    print(f"subspace_distance(U, UQ)      : "
          f"{float(eqj.subspace_distance(U, rotated)):.2e}   <- identical subspace")
    print("U and UQ describe the same subspace and give the same model, so an")
    print("elementwise comparison measures which rotation the optimiser landed")
    print("on, which is nothing. This is the same trap as the spectral and")
    print("pointwise split in the neural-operator module.")


def demo_negative_control():
    """A model with no structure must be reported as having none."""
    print("\n=== 4. the negative control ===")
    rng = np.random.default_rng(5)
    X = jnp.asarray(rng.normal(size=(400, 6)))

    isotropic = lambda Z: jnp.sum(Z ** 2, axis=1)
    ev, _ = eqj.active_subspace(isotropic, X)
    ev = np.array(ev)
    print("isotropic quadratic, eigenvalues:")
    print("  " + "  ".join("%.3f" % v for v in ev))
    print(f"ratio first / last : {ev[0] / ev[-1]:.2f}  -- no usable gap, correctly")
    print("This model varies equally in every direction, so there is no")
    print("subspace to find and the method says so. A dimension-reduction")
    print("method that always finds a subspace is not measuring anything, and")
    print("this is the check that tells the difference.")


if __name__ == "__main__":
    demo_gradient_route()
    demo_values_only_route()
    demo_identifiability()
    demo_negative_control()
