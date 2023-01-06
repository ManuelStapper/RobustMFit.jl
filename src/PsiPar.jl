# ψ Estimation
# Note: The estimation equation is solved by minimisation instead of root search

# General target function
# Working horse
function whψ(z::Vector{T1}, spec::T2)::Float64 where {T1<:Real,T2<:MSetting}
    out = 0.0
    for i = 1:length(z)
        out += ψ(z[i], spec)
    end
    out
end

# For multiple parameters
function tfψ(θ::Vector{T1},
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

    out = zeros(nPar)
    dm = dμ(dtf)

    for i = 1:nPar
        zz = (X[i, :] .- μ[i]) ./ σ[i]
        temp = whψ(zz, spec[i])
        out .+= temp .* dm[i, :]
    end

    return sum(out .^ 2)
end

# For one parameter
function tfψ(θ::T1,
    dOld::T2,
    x::Vector{T3},
    spec::T4)::Float64 where {T1<:Real,T2<:UnivariateDistribution,T3<:Real,T4<:MSetting}
    if !checkParam(dOld, θ)
        return Inf
    end

    dtf = NewDist(dOld, θ)
    μ = mean(dtf)
    σ = std(dOld)

    zz = (x .- μ) ./ σ
    return whψ(zz, spec)^2
end

# Estimation function for multiple parameters
function ψParL(d::T1,
    x::Vector{T2},
    spec::Vector{T3};
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}
    nPar = nParEff(d)
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
        θs[i+1, :] = optimize(vars -> tfψ(vars, dOld, X, specTemp), θs[i, :]).minimizer
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

function ψParL(d::T1,
    x::Vector{T2},
    spec::T3;
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}
    nPar = nParEff(d)
    # In case of multiple parameters each with the same spec
    if nPar > 1
        return ψParL(d, x, fill(spec, nPar), maxIter=maxIter, conv=conv)
    end

    θs = zeros(maxIter)
    θs[1] = getParams(d)[1]
    dOld = d

    for i = 1:maxIter-1
        specTemp = copy(spec)

        specTemp = updatekL(dOld, spec)
        θs[i+1] = optimize(vars -> vars[2]^2 + tfψ(vars[1], dOld, x, specTemp), [θs[i], 0.0]).minimizer[1]
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
# Ignored for a sec


# Estimation function for multiple parameters
function ψParU(d::T1,
    x::Vector{T2},
    spec::Vector{T3};
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}
    nPar = nParEff(d)
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
        θs[i+1, :] = optimize(vars -> tfψ(vars, dOld, X, specTemp), θs[i, :]).minimizer
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

function ψParU(d::T1,
    x::Vector{T2},
    spec::T3;
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}
    nPar = nParEff(d)
    # In case of multiple parameters each with the same spec
    if nPar > 1
        return ψParU(d, x, fill(spec, nPar), maxIter=maxIter, conv=conv)
    end

    θs = zeros(maxIter)
    θs[1] = getParams(d)[1]
    dOld = d

    for i = 1:maxIter-1
        specTemp = copy(spec)

        specTemp = updatekU(dOld, spec)
        θs[i+1] = optimize(vars -> vars[2]^2 + tfψ(vars[1], dOld, x, specTemp), [θs[i], 0.0]).minimizer[1]
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


function ψParN(d::T1,
    x::Vector{T2},
    spec::Vector{T3};
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}
    nPar = nParEff(d)
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
        θs[i+1, :] = optimize(vars -> tfψ(vars, dOld, X, specTemp), θs[i, :]).minimizer
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

function ψParN(d::T1,
    x::Vector{T2},
    spec::T3;
    maxIter::Int64=1000,
    conv::Float64=1e-05) where {T1<:UnivariateDistribution,T2<:Real,T3<:MSetting}
    nPar = nParEff(d)
    # In case of multiple parameters each with the same spec
    if nPar > 1
        return ψParN(d, x, fill(spec, nPar), maxIter=maxIter, conv=conv)
    end

    θs = zeros(maxIter)
    θs[1] = getParams(d)[1]
    dOld = d

    for i = 1:maxIter-1
        specTemp = copy(spec)

        # specTemp = updatekU(dOld, spec)
        θs[i+1] = optimize(vars -> vars[2]^2 + tfψ(vars[1], dOld, x, specTemp), [θs[i], 0.0]).minimizer[1]
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