function PTM(d::T)::Vector{Float64} where {T <: UnivariateDistribution}
    nPar = length(propertynames(d))
    out = Vector{Float64}(undef, nPar)
    for i = 1:nPar
        out[i] = mean(d^i)
    end
    return out
end

# Exceptions:

function PTM(d::BetaBinomial)::Vector{Float64}
    return [mean(d), mean(d^2)]
end

function PTM(d::Binomial)::Vector{Float64}
    return [mean(d)]
end

function PTM(d::Erlang)::Vector{Float64}
    return [mean(d)]
end

function PTM(d::NormalCanon)::Vector{Float64}
    return [mean(d), mean(d^2)]
end

function PTM(d::Pareto)::Vector{Float64}
    return [mean(d)]
end

function PTM(d::TDist)::Vector{Float64}
    return [mean(d^2)]
end

function PTM(d::VonMises)::Vector{Float64}
    return [mean(d), mean(d^2)]
end
