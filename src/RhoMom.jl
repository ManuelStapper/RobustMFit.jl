# ρ Estimation

# Working horse
function whρ(z::Vector{T1}, spec::T2) where {T1<:Real,T2<:MSetting}
    out = 0.0
    for i = 1:length(z)
        out += ρ(z[i], spec)
    end
    out
end

# General target function
function tfρ(θ::T1,
    dOld::T2,
    xp::Vector{T3},
    p::Int64,
    spec::T4)::Float64 where {T1<:Real,T2<:UnivariateDistribution,T3<:Real,T4<:MSetting}
    if iseven(p)
        if θ <= 0
            return Inf
        end
    end
    σ = std(dPower(dOld, p))
    whρ((xp .- θ) ./ σ, spec)
end

# Estimation function for multiple parameters
function ρMomL(d::T1,
    x::Vector{T2},
    spec::Vector{T3},
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}
    nPar = nParEff(d)
    θs = zeros(maxIter, nPar)
    θs[1, :] = (i -> params(d)[i]).(1:nPar)
    X = zeros(nPar, length(x))
    for i = 1:nPar
        X[i, :] = x .^ i
    end

    dOld = d
    μ = PTM(dOld)
    out = zeros(nPar)

    for i = 1:maxIter-1
        specTemp = copy(spec)

        for j = 1:nPar
            specTemp[j] = updatekL(dPower(dOld, j), spec[j])
            bounds = quantile.(dPower(dOld, j), [0.01, 0.99])
            μ[j] = optimize(vars -> tfρ(vars, dOld, X[j, :], j, specTemp[j]), bounds[1], bounds[2]).minimizer
        end
        θs[i+1, :] = MTP(μ, dOld)
        dOld = NewDist(d, θs[i+1, :])

        if checkConvergence(θs, i + 1, conv)
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

function ρMomL(d::T1,
    x::Vector{T2},
    spec::T3,
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}
    nPar = nParEff(d)
    if nPar > 1
        return ρMomL(d, x, fill(spec, nPar), maxIter, conv)
    end
    θs = zeros(maxIter)
    θs[1] = params(d)[1]

    dOld = d
    μ = PTM(dOld)[1]
    out = 0.0

    for i = 1:maxIter-1
        specTemp = updatekL(dOld, spec)
        bounds = quantile.(dOld, [0.01, 0.99])
        μ = optimize(vars -> tfρ(vars, dOld, x, 1, specTemp), bounds[1], bounds[2]).minimizer

        θs[i+1] = MTP([μ], dOld)[1]
        dOld = NewDist(d, [θs[i+1]])

        if checkConvergence(θs, i + 1, conv)
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


function ρMomU(d::T1,
    x::Vector{T2},
    spec::Vector{T3},
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}

    nPar = nParEff(d)
    θs = zeros(maxIter, nPar)
    θs[1, :] = (i -> params(d)[i]).(1:nPar)
    X = zeros(nPar, length(x))
    for i = 1:nPar
        X[i, :] = x .^ i
    end

    dOld = d
    μ = PTM(dOld)
    out = zeros(nPar)

    for i = 1:maxIter-1
        specTemp = copy(spec)

        for j = 1:nPar
            specTemp[j] = updatekU(dPower(dOld, j), spec[j])
            bounds = quantile.(dPower(dOld, j), [0.01, 0.99])
            μ[j] = optimize(vars -> tfρ(vars, dOld, X[j, :], j, specTemp[j]), bounds[1], bounds[2]).minimizer
        end
        θs[i+1, :] = MTP(μ, dOld)
        dOld = NewDist(d, θs[i+1, :])

        if checkConvergence(θs, i + 1, conv)
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


function ρMomU(d::T1,
    x::Vector{T2},
    spec::T3,
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}
    nPar = nParEff(d)
    if nPar > 1
        return ρMomU(d, x, fill(spec, nPar), maxIter, conv)
    end
    θs = zeros(maxIter)
    θs[1] = params(d)[1]

    dOld = d
    μ = PTM(dOld)[1]
    out = 0.0

    for i = 1:maxIter-1
        specTemp = updatekU(dOld, spec)
        bounds = quantile.(dOld, [0.01, 0.99])
        μ = optimize(vars -> tfρ(vars, dOld, x, 1, specTemp), bounds[1], bounds[2]).minimizer

        θs[i+1] = MTP([μ], dOld)[1]
        dOld = NewDist(d, [θs[i+1]])

        if checkConvergence(θs, i + 1, conv)
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


function ρMomN(d::T1,
    x::Vector{T2},
    spec::Vector{T3},
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}

    nPar = nParEff(d)
    θs = zeros(maxIter, nPar)
    θs[1, :] = (i -> params(d)[i]).(1:nPar)
    X = zeros(nPar, length(x))
    for i = 1:nPar
        X[i, :] = x .^ i
    end

    dOld = d
    μ = PTM(dOld)
    out = zeros(nPar)

    for i = 1:maxIter-1
        specTemp = copy(spec)

        for j = 1:nPar
            # specTemp[j] = updatekU(dPower(dOld, j), spec[j])
            bounds = quantile.(dPower(dOld, j), [0.01, 0.99])
            μ[j] = optimize(vars -> tfρ(vars, dOld, X[j, :], j, specTemp[j]), bounds[1], bounds[2]).minimizer
        end
        θs[i+1, :] = MTP(μ, dOld)
        dOld = NewDist(d, θs[i+1, :])

        if checkConvergence(θs, i + 1, conv)
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


function ρMomN(d::T1,
    x::Vector{T2},
    spec::T3,
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}
    nPar = nParEff(d)
    if nPar > 1
        return ρMomN(d, x, fill(spec, nPar), maxIter, conv)
    end
    θs = zeros(maxIter)
    θs[1] = params(d)[1]

    dOld = d
    μ = PTM(dOld)[1]
    out = 0.0

    for i = 1:maxIter-1
        # specTemp = updatekU(dOld, spec)
        bounds = quantile.(dOld, [0.01, 0.99])
        μ = optimize(vars -> tfρ(vars, dOld, x, 1, spec), bounds[1], bounds[2]).minimizer

        θs[i+1] = MTP([μ], dOld)[1]
        dOld = NewDist(d, [θs[i+1]])

        if checkConvergence(θs, i + 1, conv)
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