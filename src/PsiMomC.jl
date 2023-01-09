# ψ Estimation, old approach
# Note: The estimation equation is solved by minimisation instead of root search
#       (for computational speed reasons only)

# Working horse
function whψ(z::Vector{T1}, spec::T2, c::T3) where {T1 <: Real, T2 <: MSetting, T3 <: Real}
    out = 0.0
    for i = 1:length(z)
        out += ψ(z[i], spec)
    end
    out .- length(z)*c
end

# General target function
function tfψ(θ::T1,
             dOld::T2,
             xp::Vector{T3},
             p::Int64,
             spec::T4,
             c::T5)::Float64 where {T1 <: Real, T2 <: UnivariateDistribution, T3 <: Real, T4 <: MSetting, T5 <: Real}
    if iseven(p)
        if θ <= 0
            return Inf
        end
    end
    σ = std(dPower(dOld, p))
    whψ((xp .- θ)./σ, spec, c)^2
end

# Estimation function for multiple parameters
function ψMomC(d::T1,
                    x::Vector{T2},
                    spec::Vector{T3},
                    maxIter::Int64 = 1000,
                    conv::Float64 = 1e-05) where {T1 <: UnivariateDistribution, T2 <: Real, T3 <: MSetting}
    nPar = nParEff(d)
    θs = zeros(maxIter, nPar)
    θs[1, :] = (i -> params(d)[i]).(1:nPar)
    X = zeros(nPar, length(x))
    for i = 1:nPar
        X[i, :] = x.^i
    end

    dOld = d
    μ = PTM(dOld)
    out = zeros(nPar)
    cVec = zeros(nPar)

    for i = 1:maxIter - 1
        for j = 1:nPar
            dp = dPower(dOld, j)
            cVec[j] = findCorr(dPower(d, j), spec[j])
            bounds = quantile.(dp, [0.01, 0.99])
            μ[j] = optimize(vars -> tfψ(vars, dOld, X[j, :], j, spec[j], cVec[j]), bounds[1], bounds[2]).minimizer
        end

        θs[i+1, :] = MTP(μ, dOld)
        dOld = NewDist(d, θs[i+1, :])

        if checkConvergence(θs, i+1, conv)
            out = θs[i+1, :]
            break
        end
    end

    if out == zeros(nPar)
        @warn "Reached maximum number of iterations without convergence"
        out = θs[end, :]
    end

    return out
end

# For a single parameter
function ψMomC(d::T1,
                    x::Vector{T2},
                    spec::T3,
                    maxIter::Int64 = 1000,
                    conv::Float64 = 1e-05) where {T1 <: UnivariateDistribution, T2 <: Real, T3 <: MSetting}
    nPar = nParEff(d)
    # In case of multiple parameters each with the same spec
    if nPar > 1
        return ψMomC(d, x, fill(spec, nPar), maxIter = maxIter, conv = conv)
    end

    θs = zeros(maxIter)
    θs[1] = params(d)[1]

    dOld = d
    μ = mean(d)
    c = 0.0
    out = 0.0

    for i = 1:maxIter - 1
        # specTemp = copy(spec)
        # specTemp = updatekL(dOld, spec)
        c = findCorr(d, spec)
        bounds = quantile.(dOld, [0.01, 0.99])

        μ = optimize(vars -> tfψ(vars, dOld, x, 1, spec, c), bounds[1], bounds[2]).minimizer

        θs[i+1] = MTP([μ], dOld)[1]
        dOld = NewDist(d, [θs[i+1]])

        if checkConvergence(θs, i+1, conv)
            out = θs[i+1]
            break
        end
    end

    if out == 0.0
        @warn "Reached maximum number of iterations without convergence"
        out = θs[end]
    end

    return out
end

# Function for t Distribution
# Ignored for now
