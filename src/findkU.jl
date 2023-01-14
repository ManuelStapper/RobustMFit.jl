# Function to upper lower tuning constants for given lower tuning constants

# Idea:
# E(ψ((X-μ)/σ)) = 0

# Difference between continuous and discrete distributions
# Also between one tuning constant and multiple
# Also between smoothed and original functions

# Different target functions, depending on input

function tfkU(kk::T1,
    μ::T2,
    σ::T3,
    p::Int64,
    spec::T4,
    E::IterableExpectation{Vector{Float64},Vector{Float64}},
    target::T5) where {T1,T2,T3,T5<:Real,T4<:MSetting}
    specTF = copy(spec)
    specTF.kU = spec.kL .* kk
    E(x -> ψ((x^p - μ) / σ, specTF)) + target
end

function tfkU(kk::T1,
    μ::T2,
    σ::T3,
    p::Int64,
    spec::T4,
    sup::Vector{T5},
    prob::Vector{Float64},
    target::T6) where {T1,T2,T3,T5,T6<:Real,T4<:MSetting}
    specTF = copy(spec)
    specTF.kU = kk .* spec.kL

    out = 0.0
    for i = 1:length(prob)
        out += ψ((sup[i]^p - μ) / σ, specTF) * prob[i]
    end

    return out + target
end

"""
    findkU(d::UnivariateDistribution, spec::MSetting)
    findkU(d::dPower, spec::MSetting)
    updatekU(d::UnivariateDistribution, spec::MSetting)
    updatekU(d::dPower, spec::MSetting)

Function to find the lower tuning constant in M-estimation given the lower tuning constant.
Assuming the distribution of X is `d`, solves E(ψ(Z)) = 0 where
``Z = (X - E(X))/std(X)`` if `d` if a univariate distribution and
``Z = (X1 i - E(X^i))/std(X^i)`` if `d` if of type `dPower`.

`findkU` returns the upper tuning constant, `updatekU` returns an updated specification `spec`.

If the specification is `HampelSetting`, there is a lower tuning constant vector `kL`, not a scalar.
The function then searches for `c ⋅ kL` for suitable scalar `c`.

# Example
```julia
d = Poisson(10)
spec = Huber(0.5)
findkU(d, spec)
```

See also [`dPower`](@ref dPower).
"""
function findkU(d::T1,
    spec::T2) where {T1<:Distribution{Univariate,Continuous},T2<:MSetting}
    μ = mean(d)
    σ = std(d)

    if !isfinite(maximum(d) - minimum(d))
        E = expectation(truncated(d, quantile(d, 0.001), quantile(d, 0.999)), n=1000)
    else
        E = expectation(d, n=1000)
    end
    indNeg = E.nodes .<= μ
    Eneg = IterableExpectation(E.nodes[indNeg], E.weights[indNeg])
    Epos = IterableExpectation(E.nodes[.!indNeg], E.weights[.!indNeg])

    target = Eneg(x -> ψ((x - μ) / σ, spec))
    lower = 1e-05
    upper = 2.0
    while sign(tfkU(lower, μ, σ, 1, spec, Epos, target)) == sign(tfkU(upper, μ, σ, 1, spec, Epos, target))
        lower = copy(upper)
        upper = upper * 2
        if upper > 100
            error("Can not find suitable tuning constant")
        end
    end

    spec.kL .* find_zero(kk -> tfkU(kk, μ, σ, 1, spec, Epos, target), (lower, upper), Roots.A42(), atol=1e-05)
end

function findkU(d::T1,
    spec::T2) where {T1<:Distribution{Univariate,Discrete},T2<:MSetting}
    μ = mean(d)
    σ = std(d)
    sup = quantile(d, 0.001):quantile(d, 0.999)
    prob = pdf.(d, sup)
    indNeg = sup .<= μ
    sNeg = sup[indNeg]
    probNeg = prob[indNeg]
    sPos = sup[.!indNeg]
    probPos = prob[.!indNeg]

    target = sum(ψ.((sNeg .- μ) ./ σ, spec) .* probNeg)

    lower = 1e-05
    upper = 2.0
    while sign(tfkU(lower, μ, σ, 1, spec, sPos, probPos, target)) == sign(tfkU(upper, μ, σ, 1, spec, sPos, probPos, target))
        lower = copy(upper)
        upper = upper * 2
        if upper > 100
            error("Can not find suitable tuning constant")
        end
    end

    spec.kL .* find_zero(kk -> tfkU(kk, μ, σ, 1, spec, sPos, probPos, target), (lower, upper), Roots.A42(), atol=1e-05)
end

function findkU(d::dPower,
    spec::T) where {T<:MSetting}
    μ = mean(d)
    σ = std(d)
    p = d.p
    if iseven(p)
        cutoff = μ^(1 / p)
    else
        cutoff = sign(μ) * abs(μ)^(1 / p)
    end

    if typeof(d.d) <: Distribution{Univariate,Discrete}
        sup = quantile(d.d, 0.001):quantile(d.d, 0.999)
        prob = pdf.(d.d, sup)
        indNeg = sup .<= cutoff
        sNeg = sup[indNeg]
        probNeg = prob[indNeg]
        sPos = sup[.!indNeg]
        probPos = prob[.!indNeg]
        target = copy(sum(ψ.((sNeg .^ p .- μ) ./ σ, spec) .* probNeg))
        lower = 1e-05
        upper = 2.0
        while sign(tfkU(lower, μ, σ, p, spec, sPos, probPos, target)) == sign(tfkU(upper, μ, σ, p, spec, sPos, probPos, target))
            lower = copy(upper)
            upper = upper * 2
            if upper > 100
                error("Can not find suitable tuning constant")
            end
        end

        return spec.kL .* find_zero(kk -> tfkU(kk, μ, σ, p, spec, sPos, probPos, target), (lower, upper), Roots.A42(), atol=1e-05)
    else
        if !isfinite(maximum(d.d) - minimum(d.d))
            E = expectation(truncated(d.d, quantile(d.d, 0.001), quantile(d.d, 0.999)), n=1000)
        else
            E = expectation(d.d, n=1000)
        end
        indNeg = E.nodes .<= cutoff
        Eneg = IterableExpectation(E.nodes[indNeg], E.weights[indNeg])
        Epos = IterableExpectation(E.nodes[.!indNeg], E.weights[.!indNeg])

        target = Eneg(x -> ψ((x^p - μ) / σ, spec))
        lower = 1e-05
        upper = 2.0
        while sign(tfkU(lower, μ, σ, p, spec, Epos, target)) == sign(tfkU(upper, μ, σ, p, spec, Epos, target))
            lower = copy(upper)
            upper = upper * 2
            if upper > 100
                error("Can not find suitable tuning constant")
            end
        end

        return spec.kL .* find_zero(kk -> tfkU(kk, μ, σ, p, spec, Epos, target), (lower, upper), Roots.A42(), atol=1e-05)
    end
end

function updatekU(d::T1,
    spec::T2) where {T1<:Distribution{Univariate,Continuous},T2<:MSetting}
    kU = findkU(d, spec)
    if :ϵ in propertynames(spec)
        return typeof(spec)(spec.kL, kU, spec.ϵ)
    else
        return typeof(spec)(spec.kL, kU)
    end
end

function updatekU(d::T1,
    spec::T2) where {T1<:Distribution{Univariate,Discrete},T2<:MSetting}
    kU = findkU(d, spec)
    if :ϵ in propertynames(spec)
        return typeof(spec)(spec.kL, kU, spec.ϵ)
    else
        return typeof(spec)(spec.kL, kU)
    end
end

function updatekU(d::dPower,
    spec::T) where {T<:MSetting}
    kU = findkU(d, spec)
    if :ϵ in propertynames(spec)
        return typeof(spec)(spec.kL, kU, spec.ϵ)
    else
        return typeof(spec)(spec.kL, kU)
    end
end
