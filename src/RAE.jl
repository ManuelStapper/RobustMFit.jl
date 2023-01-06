# Compute the efficiency for given upper tuning constant (lower tc for unbiasedness)
# For multiple parameters: Single efficiencies and ignore covariances

"""
    RAE(d::UnivariateDistribution, spec::MSetting, biasCorr::Union{Symbol, String})
    RAE(d::UnivariateDistribution, spec::Vector{MSetting}, biasCorr::Union{Symbol, String})
    RAE(x::Vector{Real}, d::UnivariateDistribution, spec::MSetting, biasCorr::Union{Symbol, String})
    RAE(x::Vector{Real}, d::UnivariateDistribution, spec::Vector{MSetting}, biasCorr::Union{Symbol, String})

Relative asymptotic efficiency of M-estimator compared to Maximum Likelihood estimation.
Computation is based on [`AVar`](@ref AVar) and [`FInfo`](@ref FInfo).

If the distribution `d` has multiple parameters, the keyword argument `single` can be set to
`single = true` (default) if single variances shall be compared. `single = false' computes the p-th root
of det(Σ₁)/det(Σ₂) where Σ₁ is the covariance matrix of an ML-estimator and Σ₂ the covariance matrix
of the M-estimator.

# Example
```julia
d = Poisson(10)
x = rand(d, 200)
spec = Huber(1.5)
λ = Mfit(x, d, spec)
dFit = Poisson(λ)

RAE(dFit, spec, :L)
RAE(x, dFit, spec, :L)

d = Normal(0, 1)
x = rand(d, 200)
spec = [Huber(1.5), Huber(2.5)]
ests = Mfit(x, d, spec)
dFit = NewDist(d, ests)

RAE(dFit, spec, :L, n = 500)
RAE(dFit, spec, :L, n = 500, single = false)
RAE(x, dFit, spec, :L)
```

See [`AVar`](@ref AVar) for additional explanation of arguments.
"""
function RAE(d::Distribution{Univariate,Discrete},
        spec::Vector{T},
        biasCorr::Union{Symbol,String}=:L;
        single::Bool=true) where {T<:MSetting}
        nPar = nParEff(d)
        Σm = AVar(d, spec, biasCorr)
        Σml = inv(FInfo(d))
        if single
                return diag(Σml) ./ diag(Σm)
        else
                return (det(Σml) / det(Σm))^(1 / nPar)
        end
end

function RAE(d::Distribution{Univariate,Continuous},
        spec::Vector{T},
        biasCorr::Union{Symbol,String}=:L;
        single::Bool=true,
        n::Int64=1000) where {T<:MSetting}
        nPar = nParEff(d)
        Σm = AVar(d, spec, biasCorr, n=n)
        Σml = inv(FInfo(d))
        if single
                return diag(Σml) ./ diag(Σm)
        else
                return (det(Σml) / det(Σm))^(1 / nPar)
        end
end

function RAE(d::Distribution{Univariate, Discrete},
        spec::T,
        biasCorr::Union{Symbol,String};
        single::Bool=true) where {T<:MSetting}
        nPar = nParEff(d)
        if nPar > 1
                return RAE(d, fill(spec, nPar), biasCorr, single=single)
        end
        Σm = AVar(d, spec, biasCorr)[1, 1]
        Σml = inv(FInfo(d))
        Σml / Σm
end

function RAE(d::Distribution{Univariate, Continuous},
        spec::T,
        biasCorr::Union{Symbol,String};
        single::Bool=true,
        n::Int64 = 1000) where {T<:MSetting}
        nPar = nParEff(d)
        if nPar > 1
                return RAE(d, fill(spec, nPar), biasCorr, single=single, n = n)
        end
        Σm = AVar(d, spec, biasCorr, n = n)[1, 1]
        Σml = inv(FInfo(d))
        Σml / Σm
end


# If computed for a sample, d must be the fitted distribution (!)
function RAE(x::Vector{T1},
        d::UnivariateDistribution,
        spec::Vector{T2},
        biasCorr::Union{Symbol,String};
        single::Bool=true) where {T1<:Real,T2<:MSetting}
        nPar = nParEff(d)
        Σm = AVar(x, d, spec, biasCorr)
        Σml = inv(FInfo(d))
        if single
                return diag(Σml) ./ diag(Σm)
        else
                return (det(Σml) / det(Σm))^(1 / nPar)
        end
end

function RAE(x::Vector{T1},
        d::UnivariateDistribution,
        spec::T2,
        biasCorr::Union{Symbol,String};
        single::Bool=true) where {T1<:Real,T2<:MSetting}
        nPar = nParEff(d)
        if nPar > 1
                return RAE(x, d, fill(spec, nPar), biasCorr, single=single)
        end
        Σm = AVar(x, d, spec, biasCorr)[1, 1]
        Σml = inv(FInfo(d))
        Σml / Σm
end