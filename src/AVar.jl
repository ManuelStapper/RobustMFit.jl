# Working horse function
function whAVar(xx::T1,
                p::Int64,
                μ::Vector{Float64},
                d::T2,^
                spec::T3) where {T1 <: Real, T2 <: UnivariateDistribution, T3 <: MSetting}
    d2 = NewDist(d, MTP(μ, d))
    dp = dPower(d2, p)
    μ2 = mean(dp)

    σ = std(dp)

    z = (xx^p - μ2)/σ

    specTemp = copy(spec)
    specTemp = updatekL(dp, spec)

    ψ(z, specTemp)
end

function Eψ(sup::Vector{T1},
            prob::Vector{Float64},
            p::Int64,
            μ::Vector{Float64},
            d::T2,
            spec::T3) where {T1 <: Real, T2 <: Distribution{Univariate, Discrete}, T3 <: MSetting}
    d2 = NewDist(d, MTP(μ, d))
    dp = dPower(d2, p)
    μ2 = mean(dp)
    σ = std(dp)

    z = (sup.^p .- μ2)./σ

    specTemp = copy(spec)
    specTemp = updatekL(dp, spec)

    sum(ψ.(z, specTemp).*prob)
end

function Eψ(E::IterableExpectation,
            p::Int64,
            μ::Vector{Float64},
            d::T2,
            spec::T3) where {T1 <: Real, T2 <: Distribution{Univariate, Continuous}, T3 <: MSetting}
    d2 = NewDist(d, MTP(μ, d))
    dp = dPower(d2, p)
    μ2 = mean(dp)
    σ = std(dp)

    specTemp = copy(spec)
    specTemp = updatekL(dp, spec)

    E(xx -> ψ((xx - μ2)/σ, specTemp))
end

# Mutltiple parameters, discrete distribution
function AVar(d::T1,
              spec::Vector{T2}) where {T1 <: Distribution{Univariate, Discrete}, T2 <: MSetting}
    sup = collect(quantile(d, 0.001):quantile(d, 0.999))
    prob = pdf.(d, sup)

    nPar = nParEff(d)
    A = zeros(nPar, nPar)
    B = zeros(nPar, nPar)
    out = zeros(nPar, nPar)

    # Vector of moments
    μ = PTM(d)

    # Update the lower tuning constants
    specTemp = copy(spec)
    for i = 1:nPar
        specTemp[i] = updatekL(dPower(d, i), spec[i])
    end

    for i = 1:length(sup)
        temp = zeros(nPar)
        for j = 1:nPar
            temp[j] = whAVar(sup[i], j, μ, d, specTemp[j])
        end

        B .+= (temp*temp').*prob[i]
    end
    for i = 1:nPar
        A[i, :] = Calculus.gradient(vars -> Eψ(sup, prob, i, vars, d, spec[i]), μ)
    end

    Ainv = inv(A)
    H = MTPder(μ, d)

    return H'Ainv*B*Ainv'*H
end

# One parameter or fallback function for same specs and multiple parameters
# discrete distribution
function AVar(d::T1,
              spec::T2) where {T1 <: Distribution{Univariate, Discrete}, T2 <: MSetting}
    sup = collect(quantile(d, 0.001):quantile(d, 0.999))
    prob = pdf.(d, sup)

    nPar = nParEff(d)

    if nPar > 1
        return AVar(d, fill(spec, nPar))
    end

    B = 0.0
    out = 0.0

    # Vector of moments
    μ = PTM(d)

    # Update the lower tuning constants
    specTemp = copy(spec)
    specTemp = updatekL(d, spec)

    for i = 1:length(sup)
        B += whAVar(sup[i], 1, μ, d, specTemp)^2*prob[i]
    end

    A = Calculus.gradient(vars -> Eψ(sup, prob, 1, vars, d, spec), μ)[1]
    H = MTPder(μ, d)

    return (H.^2).*(B/A^2)
end

# Multiple parameters, continuous distribution
function AVar(d::T1,
              spec::Vector{T2}) where {T1 <: Distribution{Univariate, Continuous}, T2 <: MSetting}
    if isfinite(maximum(d) - minimum(d))
        E = expectation(d, n = 1000)
    else
        E = expectation(truncated(d, quantile(d, 0.001), quantile(d, 0.999)), n = 1000)
    end

    nPar = nParEff(d)
    A = zeros(nPar, nPar)
    B = zeros(nPar, nPar)
    out = zeros(nPar, nPar)

    # Vector of moments
    μ = PTM(d)

    # Update the lower tuning constants
    specTemp = copy(spec)
    for i = 1:nPar
        specTemp[i] = updatekL(dPower(d, i), spec[i])
    end

    for j = 1:nPar
        A[j, :] = Calculus.gradient(vars -> Eψ(E, j, vars, d, spec[j]), μ)
    end

    function f(xx, μ, d, specTemp, nPar)
        out = zeros(nPar)
        for j = 1:nPar
            out[j] = whAVar(xx, j, μ, d, specTemp[j])
        end
        return out*out'
    end

    B = E(xx -> f(xx, μ, d, specTemp, nPar))

    Ainv = inv(A)
    H = MTPder(μ, d)

    return H'Ainv*B*Ainv'*H
end

# One parameter pr mutliple parameters with same spec
# continuous distribution
function AVar(d::T1,
              spec::T2) where {T1 <: Distribution{Univariate, Continuous}, T2 <: MSetting}
    nPar = nParEff(d)
    if nPar > 1
        return AVar(d, fill(spec, nPar))
    end

    if isfinite(maximum(d) - minimum(d))
        E = expectation(d, n = 1000)
    else
        E = expectation(truncated(d, quantile(d, 0.001), quantile(d, 0.999)), n = 1000)
    end

    A = 0.0
    B = 0.0
    out = 0.0

    # Vector of moments
    μ = PTM(d)

    # Update the lower tuning constants
    specTemp = copy(spec)
    specTemp = updatekL(d, spec)

    A = Calculus.gradient(vars -> Eψ(E, 1, vars, d, spec), μ)[1]
    B = E(xx -> whAVar(xx, 1, μ, d, specTemp)^2)

    H = MTPder(μ, d)

    return (H.^2).*(B/A^2)
end

# Function for non-robust estimation
# (moments estimated by sample means)
function AVar(d::T) where {T <: UnivariateDistribution}
    nPar = nParEff(d)
    out = zeros(nPar, nPar)
    μ = (i -> mean(dPower(d, i))).(1:2*nPar)
    for i = 1:nPar, j = 1:nPar
        out[i, j] = μ[i+j] - μ[i]*μ[j]
    end
    H = MTPder(μ[1:nPar], d)
    return H'out*H
end
