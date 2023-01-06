# Functions to compute influence functions
# Given a distribution and M-Estimation specification, computes:

# -ψ(z)/E(ψ'(z)) where
# z = (x - μ)/σ
# where x is input if function being put out
# μ is the mean of the distribution
# σ the standard deviation of the distribution
"""
    IF(d::UnivariateDistribution, spec::MSetting)
    IF(d::dPower, spec::MSetting)

Influence function given the distribution `d` and a specification `spec`.
Defined as function z -> -ψ(z)/E(ψder(Z)) where
Z = (X - μ)/σ and z = (x - μ)/σ

# Example
```julia
d = Poisson(10)
spec = Huber(1.5)
IFpois = IF(d, spec)
IFpois(1)
```

See also MSetting: [`Huber`](@ref Huber), [`Tukey`](@ref Tukey), [`Andrew`](@ref Andrew) and [`Hampel`](@ref Hampel).
"""
function IF(d::T, spec::T1) where {T<:Distribution{Univariate,Continuous},T1<:MSetting}
    if isfinite(minimum(d) - maximum(d))
        E = expectation(d, n=1000)
    else
        E = expectation(truncated(d, quantile(d, 0.0001), quantile(d, 0.9999)), n=1000)
    end

    μ = mean(d)
    σ = std(d)
    c = E(x -> ψder((x - μ) / σ, spec))
    return x -> -ψ((x - μ) / σ, spec) / c
end

function IF(d::T, spec::T1) where {T<:Distribution{Univariate,Discrete},T1<:MSetting}
    sup = collect(quantile(d, 0.001):quantile(d, 0.999))

    μ = mean(d)
    σ = std(d)
    prob = pdf.(d, sup)
    c = sum(ψder.((sup .- μ) ./ σ, spec) .* prob)
    return x -> -ψ((x - μ) / σ, spec) / c
end


function IF(d::dPower, spec::T) where {T<:MSetting}
    μ = mean(d)
    σ = std(d)

    if typeof(d.d) == Distribution{Univariate,Continuous}
        if isfinite(minimum(d.d) - maximum(d.d))
            E = expectation(d.d, n=1000)
        else
            E = expectation(truncated(d.d, quantile(d.d, 0.0001), quantile(d.d, 0.9999)), n=1000)
        end
        c = E(x -> ψder((x .^ d.p - μ) / σ, spec))
    else
        sup = collect(quantile(d.d, 0.001):quantile(d.d, 0.999))
        prob = pdf.(d.d, sup)
        c = sum(ψder.((sup .^ d.p .- μ) ./ σ, spec) .* prob)
    end

    return x -> -ψ((x - μ) / σ, spec) / c
end
