# Functions that return the number of parameters to be estimated for a distirbution

# General fallback function
"""
    nParEff(d::UnivariateDistribution)
    
Get the number of parameters of a distribution `d` to be estimated.
Some parameters are treated constant.

# Example
```julia
d1 = Normal()
nParEff(d1)

d2 = Binomial(10, 0.4)
nParEff(d2)
```
"""
function nParEff(d::T)::Int64 where {T<:UnivariateDistribution}
    return length(propertynames(d))
end

function nParEff(d::BetaBinomial)::Int64
    return 2
end

function nParEff(d::Binomial)::Int64
    return 1
end

function nParEff(d::Erlang)::Int64
    return 1
end

function nParEff(d::NormalCanon)::Int64
    return 2
end

function nParEff(d::Pareto)::Int64
    return 1
end

function nParEff(d::VonMises)::Int64
    return 2
end

function nParEff(d::GeneralizedTDist)::Int64
    return 2
end
