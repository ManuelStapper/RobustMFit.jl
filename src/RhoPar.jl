# ρ Estimation

# Working horse
function whρ(z::Vector{T1}, spec::T2) where {T1<:Real,T2<:MSetting}
    out = 0.0
    for i = 1:length(z)
        out += ρ(z[i], spec)
    end
    out
end

# Target function for multiple parameters
function tfρ(θ::Vector{T1},
    dOld::T2,
    X::Matrix{T3},
    spec::Vector{T4})::Float64 where {T1<:Real,T2<:UnivariateDistribution,T3<:Real,T4<:MSetting}
    if !checkParam(dOld, θ)
        return Inf
    end
    dtf = NewDist(dOld, θ)
    nPar = nParEff(dOld)

    μ = (i -> mean(dPower(dtf, i))).(1:nPar)
    σ = (i -> std(dPower(dOld, i))).(1:nPar)

    out = 0.0
    for i = 1:nPar
        zz = (X[i, :] .- μ[i]) ./ σ[i]
        out += whρ(zz, spec[i])
    end
    out
end


# For one parameter
function tfρ(θ::T1,
    dOld::T2,
    x::Vector{T3},
    spec::T4)::Float64 where {T1<:Real,T2<:UnivariateDistribution,T3<:Real,T4<:MSetting}
    if !checkParam(dOld, θ)
        return Inf
    end
    dtf = NewDist(dOld, θ)
    μ = mean(dtf)
    σ = std(dOld)

    whρ((x .- μ) ./ σ, spec)
end

# Estimation function for multiple parameters
function ρParL(d::T1,
    x::Vector{T2},
    spec::Vector{T3},
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}
    nPar = nParEff(d)
    out = zeros(nPar)
    θs = zeros(maxIter, nPar)
    θs[1, :] = (i -> getParams(d)[i]).(1:nPar)
    X = zeros(nPar, length(x))
    for i = 1:nPar
        X[i, :] = x .^ i
    end

    dOld = d

    for i = 1:maxIter-1
        specTemp = copy(spec)

        for j = 1:nPar
            specTemp[j] = updatekL(dPower(dOld, j), spec[j])
        end
        θs[i+1, :] = optimize(vars -> tfρ(vars, dOld, X, specTemp), θs[i, :]).minimizer
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

# For one parameter
function ρParL(d::T1,
    x::Vector{T2},
    spec::T3,
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}
    nPar = nParEff(d)
    # In case of multiple parameters each with the same spec
    if nPar > 1
        return ρParL(d, x, fill(spec, nPar), maxIter=maxIter, conv=conv)
    end

    θs = zeros(maxIter)
    θs[1] = getParams(d)[1]

    dOld = d
    out = 0.0

    for i = 1:maxIter-1
        specTemp = copy(spec)
        specTemp = updatekL(dOld, spec)
        θs[i+1] = optimize(vars -> vars[2]^2 + tfρ(vars[1], dOld, x, specTemp), [θs[i], 0.0]).minimizer[1]
        dOld = NewDist(d, θs[i+1])

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
# Ignored for now



function ρParU(d::T1,
    x::Vector{T2},
    spec::Vector{T3},
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}

    nPar = nParEff(d)
    out = zeros(nPar)
    θs = zeros(maxIter, nPar)
    θs[1, :] = (i -> getParams(d)[i]).(1:nPar)
    X = zeros(nPar, length(x))
    for i = 1:nPar
        X[i, :] = x .^ i
    end

    dOld = d

    for i = 1:maxIter-1
        specTemp = copy(spec)

        for j = 1:nPar
            specTemp[j] = updatekU(dPower(dOld, j), spec[j])
        end
        θs[i+1, :] = optimize(vars -> tfρ(vars, dOld, X, specTemp), θs[i, :]).minimizer
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

# For one parameter
function ρParU(d::T1,
    x::Vector{T2},
    spec::T3,
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}
    nPar = nParEff(d)
    # In case of multiple parameters each with the same spec
    if nPar > 1
        return ρParU(d, x, fill(spec, nPar), maxIter=maxIter, conv=conv)
    end

    θs = zeros(maxIter)
    θs[1] = getParams(d)[1]
    out = 0.0

    dOld = d

    for i = 1:maxIter-1
        specTemp = copy(spec)
        specTemp = updatekU(dOld, spec)
        θs[i+1] = optimize(vars -> vars[2]^2 + tfρ(vars[1], dOld, x, specTemp), [θs[i], 0.0]).minimizer[1]
        dOld = NewDist(d, θs[i+1])

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

function ρParN(d::T1,
    x::Vector{T2},
    spec::Vector{T3},
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}

    nPar = nParEff(d)
    out = zeros(nPar)
    θs = zeros(maxIter, nPar)
    θs[1, :] = (i -> getParams(d)[i]).(1:nPar)
    X = zeros(nPar, length(x))
    for i = 1:nPar
        X[i, :] = x .^ i
    end

    dOld = d

    for i = 1:maxIter-1
        specTemp = copy(spec)

        # for j = 1:nPar
        #     specTemp[j] = updatekU(dPower(dOld, j), spec[j])
        # end
        θs[i+1, :] = optimize(vars -> tfρ(vars, dOld, X, specTemp), θs[i, :]).minimizer
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

# For one parameter
function ρParN(d::T1,
    x::Vector{T2},
    spec::T3,
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}
    nPar = nParEff(d)
    # In case of multiple parameters each with the same spec
    if nPar > 1
        return ρParN(d, x, fill(spec, nPar), maxIter=maxIter, conv=conv)
    end

    θs = zeros(maxIter)
    θs[1] = getParams(d)[1]

    dOld = d
    out = 0.0

    for i = 1:maxIter-1
        specTemp = copy(spec)
        # specTemp = updatekU(dOld, spec)
        θs[i+1] = optimize(vars -> vars[2]^2 + tfρ(vars[1], dOld, x, specTemp), [θs[i], 0.0]).minimizer[1]
        dOld = NewDist(d, θs[i+1])

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
