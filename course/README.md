# Differentiable Polynomial Approximation — a short course

A seven-lesson, hands-on introduction to approximating expensive models with
orthogonal polynomials, quantifying the uncertainty in their predictions, and —
the part that is new — **differentiating through the whole thing**.

This is grant deliverable **D5** of the NumFOCUS project *Auto-Differentiable
Equadratures*: a course on polynomial interpolation and kernel learning in
high-dimensional spaces, with a low entry barrier and a broad audience.

## Who this is for

Engineers, climate scientists and physicists who have an expensive simulation
and not enough budget to run it many times. You should be comfortable with
Python and NumPy, and remember what a mean and a variance are. **No background
in approximation theory, functional analysis or machine learning is assumed** —
each lesson introduces what it needs, in the order it needs it.

If you have never met orthogonal polynomials, start at lesson 1 and read
straight through. If you already do polynomial chaos and came for the
differentiability, skim lessons 1–4 and start properly at lesson 5. If you came
from machine learning and want the neural-operator connection, lesson 7 is
self-contained enough to read after lesson 2.

## What you will be able to do afterwards

1. Build a polynomial surrogate of a black-box model from a modest number of
   samples, and say honestly how accurate it is.
2. Choose a quadrature rule on purpose rather than by habit, and explain what
   "exact to degree 2n−1" buys you.
3. Get means, variances and Sobol' sensitivity indices out of a surrogate, and
   read the coefficients as an interpretable model rather than a black box.
4. Fit in regimes where you have fewer samples than basis terms, using sparsity.
5. Differentiate a UQ output with respect to the data, the model inputs, **and
   the assumed input distribution** — and know why that last one is useful.
6. Train a polynomial kernel, use it for Gaussian-process regression, and solve
   a Bayesian inverse problem with gradient-based MCMC.
7. Build a surrogate for an *operator* rather than a function, and say which of
   its properties are exact and which are approximations you are choosing.

## Lessons

| # | file | topic |
|---|---|---|
| 1 | `01_why_polynomials.py` | Approximating a function from samples. Interpolation, its failure modes, and why orthogonality fixes them. |
| 2 | `02_quadrature.py` | Integration you can trust. Gauss rules, exactness, and the eigenvalue problem hiding underneath. |
| 3 | `03_many_dimensions.py` | More than one input. Tensor and total-order bases, the curse of dimensionality, and sparse fits. |
| 4 | `04_uncertainty.py` | Means, variances and Sobol' indices — reading a surrogate as an interpretable model. |
| 5 | `05_gradients.py` | Differentiating through everything: data, inputs, and the input distribution itself. |
| 6 | `06_kernels_and_inference.py` | Learnable polynomial kernels, GP regression, and Bayesian inverse problems with NUTS. |
| 7 | `07_operators.py` | Learning maps between *functions*. Neural operators on a polynomial basis, and what stays exact. |

Each lesson ends with **exercises**, and each exercise has a worked solution at
the bottom of the same file, so the course works unsupervised.

## Running the course

Install what the lessons need:

```bash
pip install "equadratures[jax-learn,jax-bayes]" matplotlib
```

The lessons are plain Python scripts, so you can just run one:

```bash
python course/01_why_polynomials.py
```

They are also written in [jupytext](https://jupytext.readthedocs.io) *percent*
format, so they convert to notebooks with no loss:

```bash
pip install jupytext
jupytext --to notebook course/*.py     # produces course/*.ipynb
jupyter lab course/
```

Being ordinary Python is deliberate: it means every lesson in this course is
executed as part of development, so the code you are reading is code that runs.

Figures are written to `course/figures/` when a lesson is run as a script, and
appear inline when run as a notebook.

## A note on style

The lessons state results and then check them numerically, rather than asking
you to take them on trust. Where a method has a limitation, the lesson
demonstrates the limitation rather than avoiding an input that would expose it —
that is the honest way to teach a numerical method, and it is also how you learn
to recognise trouble in your own work.
