# Function to find lower tuning constants for given upper tuning constants

# Idea:
# E(ψ((X-μ)/σ)) = 0

# Difference between continuous and discrete distributions
# Also between one tuning constant and multiple
# Also between smoothed and original functions

# Different target functions, depending on input

function tfkL(kk::T1,
    μ::T2,
    σ::T3,
    p::Int64,
    spec::T4,
    E::IterableExpectation{Vector{Float64},Vector{Float64}},
    target::T5) where {T1,T2,T3,T5<:Real,T4<:MSetting}
    specTF = copy(spec)
    specTF.kL = spec.kU .* kk
    E(x -> ψ((x^p - μ) / σ, specTF)) + target
end

function tfkL(kk::T1,
    μ::T2,
    σ::T3,
    p::Int64,
    spec::T4,
    sup::Vector{T5},
    prob::Vector{Float64},
    target::T6) where {T1,T2,T3,T5,T6<:Real,T4<:MSetting}
    specTF = copy(spec)
    specTF.kL = kk .* spec.kU

    out = 0.0
    for i = 1:length(prob)
        out += ψ((sup[i]^p - μ) / σ, specTF) * prob[i]
    end

    return out + target
end

"""
    findkL(d::UnivariateDistribution, spec::MSetting)
    findkL(d::dPower, spec::MSetting)
    updatekL(d::UnivariateDistribution, spec::MSetting)
    updatekL(d::dPower, spec::MSetting)

Function to find the lower tuning constant in M-estimation given the upper tuning constant.
Assuming the distribution of X is `d`, solves E(ψ(Z)) = 0 where
``Z = (X - E(X))/std(X)`` if `d` if a univariate distribution and
``Z = (X1 i - E(X^i))/std(X^i)`` if `d` if of type `dPower`.

`findkL` returns the lower tuning constant, `updatekL` returns an updated specification `spec`.

If the specification is `HampelSetting`, there is an upper tuning constant vector `kU`, not a scalar.
The function then searches for `c ⋅ kU` for suitable scalar `c`.

# Example
```julia
d = Poisson(10)
spec = Huber(1.5)
findkL(d, spec)
```

See also [`dPower`](@ref dPower).
"""
function findkL(d::T1,
    spec::T2) where {T1<:Distribution{Univariate,Continuous},T2<:MSetting}
    μ = mean(d)
    σ = std(d)

    if !isfinite(maximum(d) - minimum(d))
        E = expectation(truncated(d, quantile(d, 0.001), quantile(d, 0.999)), n=10000)
    else
        E = expectation(d, n=10000)
    end
    indNeg = E.nodes .<= μ
    Eneg = IterableExpectation(E.nodes[indNeg], E.weights[indNeg])
    Epos = IterableExpectation(E.nodes[.!indNeg], E.weights[.!indNeg])

    target = Epos(x -> ψ((x - μ) / σ, spec))
    lower = 1e-05
    upper = 2.0
    while sign(tfkL(lower, μ, σ, 1, spec, Eneg, target)) == sign(tfkL(upper, μ, σ, 1, spec, Eneg, target))
        lower = copy(upper)
        upper = upper * 2
        if upper > 100
            error("Can not find suitable tuning constant")
        end
    end

    spec.kU .* find_zero(kk -> tfkL(kk, μ, σ, 1, spec, Eneg, target), (lower, upper), Roots.A42(), atol=1e-05)
end


function findkL(d::T1,
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

    target = sum(ψ.((sPos .- μ) ./ σ, spec) .* probPos)

    lower = 1e-05
    upper = 2.0
    while sign(tfkL(lower, μ, σ, 1, spec, sNeg, probNeg, target)) == sign(tfkL(upper, μ, σ, 1, spec, sNeg, probNeg, target))
        lower = copy(upper)
        upper = upper * 2
        if upper > 100
            error("Can not find suitable tuning constant")
        end
    end

    spec.kU .* find_zero(kk -> tfkL(kk, μ, σ, 1, spec, sNeg, probNeg, target), (lower, upper), Roots.A42(), atol=1e-05)
end

function findkL(d::dPower,
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
        target = sum(ψ.((sPos .^ p .- μ) ./ σ, spec) .* probPos)
        lower = 1e-05
        upper = 2.0
        while sign(tfkL(lower, μ, σ, 1, spec, sNeg, probNeg, target)) == sign(tfkL(upper, μ, σ, 1, spec, sNeg, probNeg, target))
            lower = copy(upper)
            upper = upper * 2
            if upper > 100
                error("Can not find suitable tuning constant")
            end
        end

        return spec.kU .* find_zero(kk -> tfkL(kk, μ, σ, p, spec, sNeg, probNeg, target), (lower, upper), Roots.A42(), atol=1e-05)
    else
        if !isfinite(maximum(d.d) - minimum(d.d))
            E = expectation(truncated(d.d, quantile(d.d, 0.001), quantile(d.d, 0.999)), n=10000)
        else
            E = expectation(d.d, n=10000)
        end
        indNeg = E.nodes .<= cutoff
        Eneg = IterableExpectation(E.nodes[indNeg], E.weights[indNeg])
        Epos = IterableExpectation(E.nodes[.!indNeg], E.weights[.!indNeg])

        target = Epos(x -> ψ((x^p - μ) / σ, spec))
        lower = 1e-05
        upper = 2.0
        while sign(tfkL(lower, μ, σ, 1, spec, Eneg, target)) == sign(tfkL(upper, μ, σ, 1, spec, Eneg, target))
            lower = copy(upper)
            upper = upper * 2
            if upper > 100
                error("Can not find suitable tuning constant")
            end
        end

        return spec.kU .* find_zero(kk -> tfkL(kk, μ, σ, p, spec, Eneg, target), (lower, upper), Roots.A42(), atol=1e-05)
    end
end

function updatekL(d::T1,
    spec::T2) where {T1<:Distribution{Univariate,Continuous},T2<:MSetting}
    kL = findkL(d, spec)
    if :ϵ in propertynames(spec)
        return typeof(spec)(kL, spec.kU, spec.ϵ)
    else
        return typeof(spec)(kL, spec.kU)
    end
end

function updatekL(d::T1,
    spec::T2) where {T1<:Distribution{Univariate,Discrete},T2<:MSetting}
    kL = findkL(d, spec)
    if :ϵ in propertynames(spec)
        return typeof(spec)(kL, spec.kU, spec.ϵ)
    else
        return typeof(spec)(kL, spec.kU)
    end
end

function updatekL(d::dPower,
    spec::T) where {T<:MSetting}
    kL = findkL(d, spec)
    if :ϵ in propertynames(spec)
        return typeof(spec)(kL, spec.kU, spec.ϵ)
    else
        return typeof(spec)(kL, spec.kU)
    end
end
