# Estimation of parameters with weights

# Note: Works only with raw moment estimation, then transformation to
# parameters

function whw(z::Vector{T1}, spec::T2, xp::Vector{T3}) where {T1<:Real,T2<:MSetting,T3<:Real}
    out1 = 0.0
    out2 = 0.0
    for i = 1:length(z)
        ww = w(z[i], spec)
        out1 += xp[i] * ww
        out2 += ww
    end
    return out1 / out2
end

# General function to update estimate
function tfw(dOld::T2,
    xp::Vector{T3},
    p::Int64,
    spec::T4)::Float64 where {T1<:Real,T2<:UnivariateDistribution,T3<:Real,T4<:MSetting}
    σ = std(dPower(dOld, p))
    z = (xp .- mean(dPower(dOld, p))) ./ σ
    whw(z, spec, xp)
end

# Estimation function for multiple parameters
function wMomL(d::T1,
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
    μ = zeros(nPar)
    out = zeros(nPar)

    for i = 1:maxIter-1
        specTemp = copy(spec)
        for j = 1:nPar
            specTemp[j] = updatekL(dPower(dOld, j), spec[j])
            μ[j] = tfw(dOld, X[j, :], j, specTemp[j])
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

function wMomL(d::T1,
    x::Vector{T2},
    spec::T3,
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}
    nPar = nParEff(d)
    # In case of multiple parameters each with the same spec
    if nPar > 1
        return wMomL(d, x, fill(spec, nPar), maxIter, conv)
    end

    θs = zeros(maxIter)
    θs[1] = params(d)[1]

    dOld = d
    μ = [0.0]
    out = 0.0

    for i = 1:maxIter-1
        specTemp = copy(spec)
        specTemp = updatekL(dOld, spec)

        μ[1] = tfw(dOld, x, 1, specTemp)

        θs[i+1] = MTP(μ, dOld)[1]
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

# Function for t Distribution
# Ignored


# Estimation function for multiple parameters
function wMomU(d::T1,
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
    μ = zeros(nPar)
    out = zeros(nPar)

    for i = 1:maxIter-1
        specTemp = copy(spec)
        for j = 1:nPar
            specTemp[j] = updatekU(dPower(dOld, j), spec[j])
            μ[j] = tfw(dOld, X[j, :], j, specTemp[j])
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

function wMomU(d::T1,
    x::Vector{T2},
    spec::T3,
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}
    nPar = nParEff(d)
    # In case of multiple parameters each with the same spec
    if nPar > 1
        return wMomU(d, x, fill(spec, nPar), maxIter, conv)
    end

    θs = zeros(maxIter)
    θs[1] = params(d)[1]

    dOld = d
    μ = [0.0]
    out = 0.0

    for i = 1:maxIter-1
        specTemp = copy(spec)
        specTemp = updatekU(dOld, spec)

        μ[1] = tfw(dOld, x, 1, specTemp)

        θs[i+1] = MTP(μ, dOld)[1]
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


# Estimation function for multiple parameters
function wMomN(d::T1,
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
    μ = zeros(nPar)
    out = zeros(nPar)

    for i = 1:maxIter-1
        specTemp = copy(spec)
        for j = 1:nPar
            # specTemp[j] = updatekU(dPower(dOld, j), spec[j])
            μ[j] = tfw(dOld, X[j, :], j, specTemp[j])
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

function wMomN(d::T1,
    x::Vector{T2},
    spec::T3,
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}
    nPar = nParEff(d)
    # In case of multiple parameters each with the same spec
    if nPar > 1
        return wMomN(d, x, fill(spec, nPar), maxIter, conv)
    end

    θs = zeros(maxIter)
    θs[1] = params(d)[1]

    dOld = d
    μ = [0.0]
    out = 0.0

    for i = 1:maxIter-1
        specTemp = copy(spec)
        # specTemp = updatekU(dOld, spec)

        μ[1] = tfw(dOld, x, 1, specTemp)

        θs[i+1] = MTP(μ, dOld)[1]
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