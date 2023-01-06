"""
    PTM(d::UnivariateDistribution)

Mapping parameters to moments of a unviariate distribution.
Returns the first p raw moments of the distribution, given that p
is the number of parameters to be estimated.
Relies on numerical integration of no closed form for moments is implemented.

# Example
```julia
d1 = Normal()
PTM(d1)
d2 = Binomial(10, 0.4)
PTM(d2)
```

See also [`nParEff`](@ref nParEff) for the effective number of parameters and
[`MTP`](@ref MTP) for the inverse mapping from moments to parameters. 
"""
function PTM(d::T)::Vector{Float64} where {T <: UnivariateDistribution}
    nPar = length(propertynames(d))
    out = Vector{Float64}(undef, nPar)
    for i = 1:nPar
        out[i] = mean(dPower(d, i))
    end
    return out
end

# Exceptions:

function PTM(d::BetaBinomial)::Vector{Float64}
    return [mean(d), mean(dPower(d, 2))]
end

function PTM(d::Binomial)::Vector{Float64}
    return [mean(d)]
end

function PTM(d::Erlang)::Vector{Float64}
    return [mean(d)]
end

function PTM(d::NormalCanon)::Vector{Float64}
    return [mean(d), mean(dPower(d, 2))]
end

function PTM(d::Pareto)::Vector{Float64}
    return [mean(d)]
end

function PTM(d::TDist)::Vector{Float64}
    return [mean(d^2)]
end

function PTM(d::VonMises)::Vector{Float64}
    return [mean(d), var(d)]
end
