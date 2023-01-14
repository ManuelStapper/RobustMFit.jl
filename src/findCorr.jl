# Function to compute E(ψ(z))
# Given a distribution, ψ Function with tuning constants
"""
    findCorr(d::UnivariateDistribution, spec::MSetting)
    findCorr(d::dPower, spec::MSetting)

Function to find the correction term in M-estimation.
Computes the expectation of ψ(Z) with Z = (X - μ)/σ, where X has distribution `d`.
μ and σ are the mean and standard deviation of X.
If `d` is of type `dPower` with power i, computes expectation of ψ(Z) with ``Z = (X^i - μ)/σ``
where μ and σ are the expectation and standard deviation of ``X^i``.

# Example
```julia
d = Poisson(10)
spec1 = Huber(1.5)
spec2 = Huber(2)
findCorr(d, spec1)
findCorr(dPower(d, 2), spec2)
```

See also [`dPower`](@ref dPower).
"""
function findCorr(d::T1,
    spec::T2)::Float64 where {T1<:Distribution{Univariate,Continuous},T2<:MSetting}
    μ = mean(d)
    σ = std(d)

    if !isfinite(maximum(d) - minimum(d))
        E = expectation(truncated(d, quantile(d, 0.001), quantile(d, 0.999)), n=1000)
    else
        E = expectation(d)
    end

    E(x -> ψ((x - μ) / σ, spec))
end

function findCorr(d::T1,
    spec::T2)::Float64 where {T1<:Distribution{Univariate,Discrete},T2<:MSetting}
    lb = quantile(d, 0.0001)
    ub = quantile(d, 0.9999)
    μ = mean(d)
    σ = std(d)

    sup = lb:ub
    z = (sup .- μ) ./ σ
    p = pdf.(d, sup)
    return ψ.(z, spec)'p
end

# For powers
function findCorr(d::dPower,
    spec::T)::Float64 where {T<:MSetting}
    μ = mean(d)
    σ = std(d)

    if typeof(d.d) <: Distribution{Univariate,Continuous}
        if !isfinite(maximum(d.d) - minimum(d.d))
            E = expectation(truncated(d.d, quantile(d.d, 0.001), quantile(d.d, 0.999)), n=1000)
        else
            E = expectation(d.d)
        end

        E(x -> ψ((x^d.p - μ) / σ, spec))
    else
        sup = collect(quantile(d.d, 0.001):quantile(d.d, 0.999))
        prob = pdf.(d.d, sup)
        return sum(ψ.((sup .^ d.p .- μ) ./ σ, spec) .* prob)
    end
end
