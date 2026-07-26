"""Bayesian inverse problems on top of a differentiable polynomial surrogate.

This is the deliverable that only exists *because* the surrogate is
differentiable. Gradient-based samplers -- NUTS, and stochastic variational
inference -- need ``d(log density)/d(parameters)`` through the forward model.
With the classic NumPy ``equadratures`` that derivative is unavailable, so
inference has to fall back on random-walk Metropolis or on finite differences,
both of which scale badly in the number of unknowns. With
:class:`equadratures.jax.poly.Poly` the forward model is an ordinary JAX
function, so NumPyro can differentiate straight through it.

Two problems are provided, covering the two directions people actually want:

* :func:`surrogate_input_model` -- **inverse UQ**. Given noisy observations of a
  system's output, infer the *inputs* that produced them. The surrogate stands
  in for an expensive simulation, and NUTS explores the input posterior.
* :func:`sparse_coefficient_model` -- **Bayesian compressed sensing**. Infer the
  polynomial *coefficients* under a sparsity-inducing prior. This is the
  probabilistic counterpart of :func:`equadratures.jax.solver.lasso`: a Laplace
  prior recovers the ``l1`` solve at the posterior mode, and the regularised
  horseshoe used here does the same job while also reporting which coefficients
  it is actually confident about.

NumPyro and ArviZ are optional; install them with::

    pip install equadratures[jax-bayes]

This module deliberately is *not* imported by ``equadratures.jax.__init__``, so
the base namespace never pulls in NumPyro. Import it explicitly::

    from equadratures.jax.inverse import surrogate_input_model, run_nuts
"""
import jax
import jax.numpy as jnp

try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
    from numpyro.infer.autoguide import AutoNormal
    _HAS_NUMPYRO = True
except ImportError:                                  # pragma: no cover
    _HAS_NUMPYRO = False


def _require_numpyro():
    if not _HAS_NUMPYRO:                             # pragma: no cover
        raise ImportError(
            "NumPyro is required for equadratures.jax.inverse. "
            "Install it with: pip install equadratures[jax-bayes]")


# ------------------------------------------------------------------- models
def surrogate_input_model(forward, y_obs, lower, upper, noise_scale=None):
    """NumPyro model inferring model *inputs* from noisy output observations.

    Parameters
    ----------
    forward : callable
        ``x -> predicted observations``, where ``x`` has shape ``(d,)`` and the
        result has the same shape as ``y_obs``. Must be JAX-traceable and
        differentiable -- typically a closure over a fitted
        :class:`equadratures.jax.poly.Poly`.
    y_obs : array_like
        Observed outputs.
    lower, upper : array_like, shape (d,)
        Support of the uniform prior on the inputs. For a surrogate built on a
        standard domain these are the domain bounds, which keeps the sampler
        inside the region where the polynomial is trustworthy.
    noise_scale : float or None
        Observation noise standard deviation. If ``None`` it is inferred too,
        under a half-normal prior.

    Returns
    -------
    callable
        A NumPyro model, ready for :func:`run_nuts` or :func:`run_svi`.

    Notes
    -----
    Sampling ``x`` under a bounded prior means NUTS works in an unconstrained
    reparameterisation; NumPyro applies and accounts for that transform itself.
    """
    _require_numpyro()
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)
    y_obs = jnp.asarray(y_obs)

    def model():
        x = numpyro.sample("x", dist.Uniform(lower, upper).to_event(1))
        if noise_scale is None:
            sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        else:
            sigma = noise_scale
        numpyro.sample("y", dist.Normal(forward(x), sigma), obs=y_obs)

    return model


def sparse_coefficient_model(A, y, tau_scale=0.1, noise_scale=None):
    """NumPyro model for polynomial coefficients under a sparsity prior.

    The Bayesian counterpart of the ``l1`` solve in
    :mod:`equadratures.jax.solver`. A horseshoe prior puts most coefficients
    near zero while leaving heavy tails so genuinely large ones escape
    shrinkage -- the behaviour ``l1`` achieves by thresholding, but with a
    posterior distribution attached, so "this term is zero" becomes a statement
    with a credible interval rather than an artefact of the penalty.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Design matrix.
    y : array_like, shape (m,)
        Observations.
    tau_scale : float
        Scale of the global shrinkage parameter. Smaller means sparser.
    noise_scale : float or None
        Observation noise. Inferred under a half-normal prior when ``None``.
    """
    _require_numpyro()
    A = jnp.asarray(A)
    y = jnp.asarray(y)
    n = A.shape[1]

    def model():
        tau = numpyro.sample("tau", dist.HalfCauchy(tau_scale))
        lam = numpyro.sample("lam", dist.HalfCauchy(jnp.ones(n)).to_event(1))
        c = numpyro.sample("c", dist.Normal(jnp.zeros(n), tau * lam).to_event(1))
        if noise_scale is None:
            sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        else:
            sigma = noise_scale
        numpyro.sample("y", dist.Normal(A @ c, sigma), obs=y)

    return model


# ------------------------------------------------------------------ runners
def run_nuts(model, num_warmup=500, num_samples=1000, num_chains=1, seed=0,
             progress_bar=False, **nuts_kwargs):
    """Run the No-U-Turn sampler on ``model`` and return the fitted ``MCMC``.

    NUTS is gradient-based: every leapfrog step differentiates the log density,
    and hence the surrogate. That is exactly the capability the JAX namespace
    adds.

    Notes
    -----
    On a single CPU device ``num_chains > 1`` runs the chains sequentially and
    NumPyro warns as much. To get them in parallel, call
    ``numpyro.set_host_device_count(n)`` before any JAX work happens. Two chains
    are worth the wait regardless, since ``r_hat`` needs more than one.
    """
    _require_numpyro()
    kernel = NUTS(model, **nuts_kwargs)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=progress_bar)
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc


def run_svi(model, num_steps=2000, learning_rate=1e-2, seed=0):
    """Fit a mean-field normal guide by stochastic variational inference.

    Cheaper than MCMC and often enough for a first look; also gradient-based,
    so it has the same requirement on the forward model.

    Returns
    -------
    guide : AutoNormal
    result : SVIRunResult
        ``result.losses`` is the ELBO trace, ``result.params`` the fitted guide
        parameters.
    """
    _require_numpyro()
    import optax

    guide = AutoNormal(model)
    svi = SVI(model, guide, optax.adam(learning_rate), Trace_ELBO())
    result = svi.run(jax.random.PRNGKey(seed), num_steps, progress_bar=False)
    return guide, result


def posterior_samples(mcmc, site=None):
    """Posterior samples as plain arrays, optionally for one site only."""
    samples = mcmc.get_samples()
    return samples if site is None else samples[site]


def summarise(mcmc):
    """Posterior mean / std / quantiles per site, as a dict of dicts."""
    out = {}
    for name, draws in mcmc.get_samples().items():
        arr = jnp.asarray(draws)
        out[name] = {
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0),
            "q05": jnp.quantile(arr, 0.05, axis=0),
            "q95": jnp.quantile(arr, 0.95, axis=0),
        }
    return out


def to_arviz(mcmc):
    """Convert to an ArviZ ``InferenceData`` for diagnostics and plotting.

    Gives access to ``r_hat``, effective sample size, trace plots and the rest
    of the standard convergence toolkit.
    """
    try:
        import arviz as az
    except ImportError:                              # pragma: no cover
        raise ImportError(
            "ArviZ is required for to_arviz. "
            "Install it with: pip install equadratures[jax-bayes]")
    return az.from_numpyro(mcmc)


def predictive_samples(model, mcmc, seed=0, **model_kwargs):
    """Posterior predictive draws from a fitted model."""
    _require_numpyro()
    predictive = Predictive(model, mcmc.get_samples())
    return predictive(jax.random.PRNGKey(seed), **model_kwargs)
