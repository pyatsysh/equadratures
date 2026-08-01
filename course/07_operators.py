# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lesson 7 — Learning operators, not just functions
#
# Every lesson so far has approximated a **function**: give it numbers, get a
# number back. This one approximates an **operator**: give it a *whole function*,
# get a whole function back.
#
# That sounds abstract until you notice how many expensive simulations are
# actually operators. A solver that takes an initial temperature *profile* and
# returns the profile an hour later. One that takes a spatially varying material
# property and returns the resulting stress field. A weather model taking today's
# state to tomorrow's. In each case the input is not a handful of parameters but
# a function, and the thing you would like to replace with a surrogate is the map
# between functions.
#
# This is what **neural operators** do, and the reason they belong in this course
# is that the standard recipe has a polynomial version which is better behaved in
# three specific ways — and honest about where it is not.

# %%
import os

import numpy as np
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equadratures.jax as eqj
from equadratures.jax.basis import design_matrix

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def save(name):
    plt.savefig(os.path.join(FIGDIR, name), dpi=120, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## 7.1 The shape of every neural operator
#
# There is really only one architecture in this field. A layer is
#
# $$v(x) = \sigma\big( W u(x) + (K u)(x) + b \big), \qquad
#   (K u)(x) = \int \kappa(x, y)\, u(y)\, dy$$
#
# A pointwise part `W u(x) + b`, which mixes channels at each point and cannot
# move information *between* points; an **integral operator** `K`, which is the
# only part that can; and a nonlinearity. Published architectures differ in
# exactly one respect: how they represent and apply `K`.
#
# The Fourier neural operator makes $\kappa$ diagonal in a Fourier basis and
# applies it with an FFT. We already have a better-suited basis — the orthonormal
# polynomials this whole course is built on — and a way to integrate exactly, so:
#
# $$\kappa(x, y) = \sum_{j,k} R_{jk}\, \phi_j(x)\, \phi_k(y), \qquad
#   (K u)(x) = \sum_{j,k} R_{jk}\, \phi_j(x)\, c_k$$
#
# where $c_k = \sum_i w_i \phi_k(x_i) u(x_i)$ is the spectral projection from
# lesson 2. Three steps: **project** the input onto polynomial modes, **mix** the
# modes with a matrix `R`, **reconstruct**. The integral never appears at run
# time; it has been done exactly, in advance, by the quadrature.

# %%
ORDER = 6
recurrences = [eqj.uniform_recurrence(ORDER + 4)]
indices = eqj.total_order_indices(1, ORDER)
layer = eqj.SpectralOperatorLayer(recurrences, indices)

print(f"polynomial modes : {layer.n_modes}")
print(f"quadrature nodes : {layer.X.shape[0]}")
print(f"spectral tensor R: {layer.spectral_shape()}   (modes x modes x c_out x c_in)")

# %% [markdown]
# ## 7.2 An operator you can write down
#
# Before learning anything, notice that some operators need no learning at all.
# If `A` is linear and maps the polynomial span into itself, its matrix in this
# basis is just
#
# $$R_{jk} = \langle \phi_j, A\phi_k \rangle$$
#
# and that is an integral we can do exactly. **Differentiation** is the obvious
# case: it takes a polynomial of degree $p$ to one of degree $p-1$, which is
# still in the span.

# %%
R_ddx = eqj.derivative_operator_tensor(layer)

coeffs = jax.random.normal(jax.random.PRNGKey(1), (layer.n_modes, 1))
u = layer.evaluate(coeffs)                       # a function that IS in the span

got = layer.integral(u, R_ddx)[:, 0]
scalar = lambda x: (layer.design(x[None, :]) @ coeffs)[0, 0]
want = jax.vmap(jax.grad(scalar))(layer.X)[:, 0]

err = float(jnp.abs(got - want).max())
print(f"max |K u  -  du/dx|           : {err:.3e}")
print(f"relative to the size of du/dx : {err / float(jnp.abs(want).max()):.3e}")

# %% [markdown]
# That is machine precision. Not "a good approximation to the derivative" — the
# *exact* derivative, recovered by a matrix multiply, because differentiation
# genuinely is a finite matrix in this basis.
#
# Worth pausing on: differentiation is **not symmetric**, and a kernel that is
# diagonal in the modes is automatically symmetric. So a mode-diagonal layer —
# the direct analogue of the FNO — *cannot* represent it, no matter how it is
# trained. That is why the dense parameterisation exists.

# %%
Xq = jnp.linspace(-1, 1, 400).reshape(-1, 1)
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
ax[0].plot(np.array(Xq)[:, 0], np.array(layer.design(Xq) @ coeffs)[:, 0], lw=2)
ax[0].set_title("input function $u$"); ax[0].grid(alpha=.3)
ax[1].plot(np.array(Xq)[:, 0], np.array(layer.integral(u, R_ddx, Xq))[:, 0],
           lw=2, color="C1", label="$K u$ (operator)")
ax[1].plot(np.array(layer.X)[:, 0], np.array(want), "k.", ms=9,
           label="autodiff $du/dx$")
ax[1].set_title("output: an operator applied, not a curve fitted")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
save("07_operator_action.png")

# %% [markdown]
# ## 7.3 Why this is an *operator* method: discretisation invariance
#
# Here is the property that separates an operator surrogate from an ordinary
# regression on grid values. The parameters `R` live on **polynomial modes**, not
# on grid points. So the same trained layer can be evaluated on a completely
# different quadrature and returns *the same function*.
#
# This is the headline claim of the neural-operator literature, where it holds
# approximately. Here it holds to round-off, which is a different kind of
# statement.

# %%
coarse = eqj.SpectralOperatorLayer([eqj.uniform_recurrence(ORDER + 2)], indices)
fine = eqj.SpectralOperatorLayer([eqj.uniform_recurrence(ORDER + 12)], indices)
R_c = eqj.derivative_operator_tensor(coarse)

c_test = jax.random.normal(jax.random.PRNGKey(2), (coarse.n_modes, 1))
out_coarse = coarse.integral(coarse.evaluate(c_test), R_c, Xq)
out_fine = fine.integral(fine.evaluate(c_test), R_c, Xq)

print(f"coarse rule : {coarse.X.shape[0]:2d} nodes")
print(f"fine rule   : {fine.X.shape[0]:2d} nodes")
print(f"max difference over 400 query points: "
      f"{float(jnp.abs(out_coarse - out_fine).max()):.3e}")

plt.figure(figsize=(6.4, 3.4))
plt.plot(np.array(Xq)[:, 0], np.array(out_coarse)[:, 0], lw=3, alpha=.5,
         label=f"{coarse.X.shape[0]} nodes")
plt.plot(np.array(Xq)[:, 0], np.array(out_fine)[:, 0], "--", lw=1.5,
         label=f"{fine.X.shape[0]} nodes")
plt.legend(); plt.grid(alpha=.3)
plt.title("same parameters, two discretisations, one function")
save("07_discretisation.png")

# %% [markdown]
# ## 7.4 Learning an operator from data
#
# Now the part that is actually "neural". Suppose you do *not* know the operator,
# only pairs: input functions and what the expensive solver returned for them. We
# will use $d/dx$ as the hidden truth precisely because we can check the answer.

# %%
n_train = 64
train_coeffs = jax.random.normal(jax.random.PRNGKey(3), (n_train, layer.n_modes, 1))
U = jax.vmap(layer.evaluate)(train_coeffs)                  # input functions
Y = jax.vmap(lambda f: layer.integral(f, R_ddx))(U)         # what the solver gave


def training_loss(params):
    predictions = jax.vmap(lambda f: layer.apply(f, params))(U)
    return jnp.mean((predictions - Y) ** 2)


# %% [markdown]
# One detail that is not a detail. The learning rate is **annealed** rather than
# held fixed, and the reason is worth knowing before it bites you: Adam divides
# the gradient by $\sqrt{v}$, so once the gradient reaches round-off it is
# dividing noise by noise and *amplifies* it. Written with a constant rate, this
# exact demonstration reached a loss of 8e-21 by step 3000 and bounced back to
# 1e-5 by step 4000. Annealing costs a few digits and buys a result that does not
# depend on the seed.

# %%
try:
    import optax
    HAVE_OPTAX = True
except ImportError:
    HAVE_OPTAX = False
    print("optax not installed; skipping training "
          "(pip install equadratures[jax-learn])")

if HAVE_OPTAX:
    STEPS = 6000
    params = layer.init_params(jax.random.PRNGKey(4))
    schedule = optax.cosine_decay_schedule(3e-2, decay_steps=STEPS, alpha=1e-4)
    opt = optax.adam(schedule)
    state = opt.init(params)
    step_fn = jax.jit(jax.value_and_grad(training_loss))

    history = []
    for it in range(STEPS):
        value, grads = step_fn(params)
        history.append(float(value))
        if it % 1200 == 0:
            print(f"   step {it:4d}: loss = {float(value):.3e}")
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)
    print(f"   final    : loss = {float(training_loss(params)):.3e}")

    plt.figure(figsize=(6, 3.2))
    plt.semilogy(history)
    plt.xlabel("Adam step"); plt.ylabel("training loss")
    plt.grid(alpha=.3); plt.title("learning an operator from 64 function pairs")
    save("07_training.png")

# %% [markdown]
# ## 7.5 Did it learn the operator, or just the data?
#
# A small loss is not the claim worth making. The claim worth making is that the
# **operator itself** was recovered — and checking that exposes a trap.
#
# The obvious check is to compare the learned `params["spectral"]` against the
# analytic `R`. That comparison is **meaningless**, and it is worth seeing why
# before seeing it happen, because the demonstration does not need training at
# all — it can be constructed.
#
# The pointwise term `W u(x)` is *itself* representable in the spectral tensor:
# on the polynomial representation it acts as `W` times the identity in mode
# space. So take the exact operator, move a constant `s` out of the spectral
# tensor and into `W`, and you have a completely different parameter set that
# computes the identical function.

# %%
print("the same operator, split three different ways:")
print(f"  {'shift s':>8} {'prediction error':>18} {'|raw R - R_true|':>18} "
      f"{'|effective - R_true|':>22}")
for s in (0.0, 2.0, -5.0):
    split = {"spectral": R_ddx - s * jnp.eye(layer.n_modes)[:, :, None, None],
             "pointwise": jnp.full((1, 1), s),
             "bias": jnp.zeros(1)}
    pred_err = float(jnp.abs(layer.apply(u, split) - layer.integral(u, R_ddx)).max())
    raw = float(jnp.abs(split["spectral"] - R_ddx).max())
    eff = float(jnp.abs(eqj.effective_spectral_tensor(layer, split) - R_ddx).max())
    print(f"  {s:8.1f} {pred_err:18.2e} {raw:18.3f} {eff:22.2e}")

# %% [markdown]
# Read the columns. Every row is the *same operator* — the predictions agree to
# round-off. But the raw spectral tensor is off by exactly `s`, which we chose
# arbitrarily. So `spectral` and `pointwise` are **not separately identifiable**:
# infinitely many splits give the identical operator, and which one an optimiser
# lands on is an accident of initialisation.
#
# `effective_spectral_tensor` combines them, and its column is right in every
# row. That is what to compare. Now the trained layer, with the same test:

# %%
if HAVE_OPTAX:
    raw_gap = float(jnp.abs(params["spectral"] - R_ddx).max())
    effective = eqj.effective_spectral_tensor(layer, params)
    eff_gap = float(jnp.abs(effective - R_ddx).max())
    scale = float(jnp.abs(R_ddx).max())

    print(f"|raw spectral      - R_true|_max : {raw_gap:.3e}   <- measures W")
    print(f"|effective operator- R_true|_max : {eff_gap:.3e}   <- measures the fit")
    print(f"(for scale, |R_true|_max = {scale:.3f})")

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    vmax = float(jnp.abs(R_ddx).max())
    for a, M, t in ((ax[0], np.array(R_ddx)[:, :, 0, 0], "true $R$ for $d/dx$"),
                    (ax[1], np.array(effective)[:, :, 0, 0], "learned (effective)"),
                    (ax[2], np.array(params["spectral"])[:, :, 0, 0], "learned (raw)")):
        im = a.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_title(t, fontsize=10); a.set_xlabel("input mode $k$")
    ax[0].set_ylabel("output mode $j$")
    fig.colorbar(im, ax=ax, shrink=.85)
    save("07_learned_operator.png")

# %% [markdown]
# Look at the left panel for a moment, because this is the interpretability
# argument made concrete. `R` for $d/dx$ is **strictly lower-triangular in
# degree**: differentiating mode $k$ can only produce modes of lower degree. You
# can read that off the picture. An FNO's learned weights are complex Fourier
# multipliers; there is no comparable picture to read.

# %% [markdown]
# ## 7.6 Where it stops being exact
#
# Two limits, stated plainly rather than discovered later.
#
# **Nonlinearity truncates.** Between layers the output is re-projected onto the
# polynomial span, and $\sigma(v)$ is generally not in the span even when $v$ is.
# So a *linear* stack composes exactly; a nonlinear one carries a truncation
# error at every layer, controlled by the mode count. This is not special to
# polynomials — the FNO truncates in exactly the same place — but it does mean
# the exactness results above belong to the linear case.
#
# **The input must be representable.** If the function you feed in is not well
# approximated by the basis, the operator is applied faithfully to the *wrong
# input*. That error is a property of lesson 1, not of this lesson, and it
# behaves exactly as lesson 1 said it would.

# %%
stack = eqj.NeuralOperator([layer, layer], activation=None)
stack_params = stack.init_params(jax.random.PRNGKey(5))
u2 = layer.evaluate(jax.random.normal(jax.random.PRNGKey(6), (layer.n_modes, 1)))
a, b = 0.3, -1.7
zero = stack.apply(0.0 * u, stack_params)
lhs = stack.apply(a * u + b * u2, stack_params) - zero
rhs = (a * (stack.apply(u, stack_params) - zero)
       + b * (stack.apply(u2, stack_params) - zero))
print(f"linear stack, superposition error: {float(jnp.abs(lhs - rhs).max()):.3e}")
print("A linear stack of exact operators is an exact operator.")

# %% [markdown]
# ## What to take away
#
# * An **operator** maps functions to functions; many expensive simulations are
#   operators, and that is what you want to surrogate.
# * Every neural operator is `pointwise + integral operator + nonlinearity`. The
#   only real design choice is how the integral operator is represented.
# * Representing it in an **orthonormal polynomial basis** and applying it by
#   **quadrature** gives three things: the parameters are readable as an
#   operator's matrix on known modes; any linear operator that maps the span into
#   itself is represented *exactly*; and discretisation invariance is exact
#   rather than approximate.
# * A linear operator you can write down needs **no training at all**.
# * When checking a learned operator, compare the **effective** tensor. Raw
#   parameter blocks are not identifiable, and comparing them measures the
#   initialisation.
# * Anneal the learning rate. Adam is unstable once the gradient is at round-off.
#
# ## Where to go next
#
# * `examples/jax_neural_operator.py` — the condensed runnable version.
# * `equadratures/jax/README.md` section 6 — the API reference.
# * `tests/test_jax_operator.py` — every claim in this lesson, as an assertion.
#
# ## Exercises
#
# 1. Build the operator for the **second** derivative and check it is exact too.
#    Which degrees can it not represent, and why is that not a problem here?
# 2. Feed the layer a function that is *not* in the span — `|x|`, and then
#    `exp(x)` — and measure the error in `K u`. Does refining the polynomial
#    order fix it in both cases at the same rate? You met the answer in lesson 1.
# 3. Take the trained layer from Section 7.4 and evaluate it on a quadrature it
#    never saw. Then do the same for a *nonlinear* two-layer stack. Which one is
#    still discretisation-invariant, and why?
#
# Solutions below.

# %%
if __name__ == "__main__":
    print("\n--- Solutions ---")

    # 1. d2/dx2 maps degree p to p-2, so it also maps the span into itself.
    def second_derivative_basis(X):
        def row(x):
            return design_matrix(x[None, :], layer.indices, layer.recurrences)[0]
        return jax.vmap(jax.jacfwd(jax.jacfwd(row)))(X)[:, :, 0, 0]

    R_d2 = eqj.linear_operator_tensor(layer, second_derivative_basis)
    got2 = layer.integral(u, R_d2)[:, 0]
    want2 = jax.vmap(jax.jacfwd(jax.jacfwd(scalar)))(layer.X)[:, 0, 0]
    e2 = float(jnp.abs(got2 - want2).max())
    print(f"1. max |K u - d2u/dx2| = {e2:.2e}  "
          f"(relative {e2 / float(jnp.abs(want2).max()):.1e})")
    print("   Exact. It cannot produce the top two degrees -- differentiating")
    print("   twice always loses two -- but it never needs to, because the")
    print("   OUTPUT of d2/dx2 on a degree-p input is only degree p-2.")

    # 2. Off-span inputs: the operator is fine, the representation is not.
    #    Two norms, because they say different things here.
    print("2. applying the exact d/dx operator to functions outside the span:")
    print(f"   {'order':>6} {'|x| RMS':>11} {'|x| max':>11} "
          f"{'exp RMS':>11} {'exp max':>11}")
    for p in (4, 8, 16, 24, 32):
        Lp = eqj.SpectralOperatorLayer([eqj.uniform_recurrence(p + 4)],
                                       eqj.total_order_indices(1, p))
        Rp = eqj.derivative_operator_tensor(Lp)
        row = []
        for f, df in ((jnp.abs, lambda x: jnp.sign(x)), (jnp.exp, jnp.exp)):
            e = Lp.integral(f(Lp.X[:, :1]), Rp, Xq)[:, 0] - df(Xq[:, 0])
            row += [float(jnp.sqrt(jnp.mean(e ** 2))), float(jnp.abs(e).max())]
        print(f"   {p:6d} {row[0]:11.3e} {row[1]:11.3e} "
              f"{row[2]:11.3e} {row[3]:11.3e}")
    print("   The operator is identical in every row -- it is exact. What")
    print("   differs is whether the INPUT can be represented, and the two")
    print("   norms disagree in an instructive way.")
    print("   exp(x): exponential convergence, to a floor near 1e-12. Past")
    print("   order 16 it gets slightly WORSE, because a very high degree")
    print("   basis is more poorly conditioned -- more resolution is not free.")
    print("   |x|: the max error never converges at all. d|x|/dx has a jump,")
    print("   and no polynomial resolves a jump -- the overshoot beside it")
    print("   stays put and only narrows. The RMS does fall, but algebraically")
    print("   (about 0.40 to 0.15 for an eightfold order increase). Lesson 1's")
    print("   Runge warning, arriving again one level up.")

    # 3. Discretisation invariance survives linearity, not nonlinearity.
    if HAVE_OPTAX:
        other = eqj.SpectralOperatorLayer(
            [eqj.uniform_recurrence(ORDER + 11)], indices)
        c_new = jax.random.normal(jax.random.PRNGKey(7), (layer.n_modes, 1))
        lin_a = layer.apply(layer.evaluate(c_new), params, Xq)
        lin_b = other.apply(other.evaluate(c_new), params, Xq)
        print(f"3. trained single layer, two rules : "
              f"{float(jnp.abs(lin_a - lin_b).max()):.2e}")

        nl_a = eqj.NeuralOperator([layer, layer], activation=jax.nn.tanh)
        nl_b = eqj.NeuralOperator([other, other], activation=jax.nn.tanh)
        p_nl = nl_a.init_params(jax.random.PRNGKey(8))
        d_nl = float(jnp.abs(nl_a.apply(layer.evaluate(c_new), p_nl, Xq)
                             - nl_b.apply(other.evaluate(c_new), p_nl, Xq)).max())
        print(f"   nonlinear two-layer stack        : {d_nl:.2e}")
        print("   The linear layer is invariant to round-off. The nonlinear")
        print("   stack is not: tanh(v) leaves the polynomial span, so each")
        print("   layer re-projects, and how much is lost depends on the rule")
        print("   doing the projecting. Invariance is a property of the linear")
        print("   case, and it is worth knowing which half of your architecture")
        print("   still has it.")
