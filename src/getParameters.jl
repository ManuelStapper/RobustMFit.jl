
"""
    getParams(d::UnivariateDistribution)
    
Selects the parameters of a distribution and gives them out as vector.
Only parameters to be estimated are returned.

# Example
```julia
d1 = Normal()
getParams(d1)

d2 = Binomial(10, 0.4)
getParams(d2)
```

See also [`nParEff`](@ref nParEff).
"""
function getParams(d::UnivariateDistribution)
    pRaw = params(d)
    if length(pRaw) == 1
        return pRaw[1]
    else
        (i -> pRaw[i]).(1:length(pRaw))
    end

end

function getParams(d::Binomial)
    [params(d)[2]]
end

# Exceptions for distributions with fixed parameters
function getParams(d::BetaBinomial)
    pRaw = params(d)
    return [pRaw[2], pRaw[3]]
end

function getParams(d::Binomial)
    [params(d)[2]]
end

function getParams(d::Erlang)
    [params(d)[2]]
end

function getParams(d::Pareto)
    [params(d)[2]]
end

function getParams(d::GeneralizedTDist)
    params(d)[1:2]
end
