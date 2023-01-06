MFit.jl
=======

[![Build status](https://ci.appveyor.com/api/projects/status/2rir1d6va30gw3hx?svg=true)](https://ci.appveyor.com/project/ManuelStapper/mfit-jl)
[![codecov](https://codecov.io/gh/ManuelStapper/MFit.jl/branch/main/graph/badge.svg?token=3WH9IXRGBC)](https://codecov.io/gh/ManuelStapper/MFit.jl)

A Julia package to estimate parameters of a univariate distribution robustly by M-estimation. Currently implemented are Tukey, Huber, Andrew and Hampel functions as well as smoothed versions of the latter three. User-defined functions can be added using the `NewMFunction.jl` template.

Different estimation approaches are implemented:

## Function types

Estimation can be carried out by minimizing a loss function ($$\rho$$-type), finding the root of its derivative ($$\psi$$-type) or iteratively by a weight function ($$w$$-type).

## Method of Moments

The parameters can be estimated either directly or by estimating the raw moments of the distribution and translating them to parameters.

## Bias Correction

Estimation is only unbiased in general if the underlying distribution is symmetric. A potential bias must also be taken into account for symmetric distributions with multiple parameters. Esimating for example the mean of a Normal distribution with symmetric ($$\rho$$, $$\psi$$ or $$w$$)-function is unproblematic, but estimating the variance parameter is not, since the distribution of $$X^2$$ is asymmetric.

The bias can be tackled by a correction term in the $$\psi$$ estimation or by using asymmetric ($$\rho$$, $$\psi$$ or $$w$$)-functions. For asymmetric functions, different tuning constants are selected for positive and negative input. One of those is kept constant while the other is chosen such that the estimator is consistent.

# Examples

```julia
d = Poisson(10)
x = rand(d, 100)

# For ρ-Esimation, Moment based and updating lower tuning constant
λ = Mfit(x, d, Huber(1.5), type=:ρ, MM=true, biasCorr=:L)

# ψ-Estimation, estimate parameters directly, update upper tuning constant
Mfit(x, d, Tukey(4), type = :ψ, MM = false, biasCorr = :U)
# w-Estimation, Moment based, not accounting for asymmetry at all
Mfit(x, d, Tukey(4), type = :ψ, MM = false, biasCorr = :N)
# Or the same, but accounting for it with correction term
Mfit(x, d, Tukey(4), type = :ψ, MM = false, biasCorr = :C)

# Computing the asymptotic variance of the first estimation
AVar(Poisson(λ), Huber(1.5), :L)
# Or estimating it using the sample
AVar(x, Poisso(λ), Huber(1.5), :L)

# Comparing the robust estaimtion with ML estimation by relative asymptotic efficiency
RAE(Poisson(λ), Huber(1.5), :L)
```

