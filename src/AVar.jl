# Define target functions (ψ functions)

# If lower tuning constant is updated
function tfAVarL(θ::Vector{T1},
    d::T2,
    i::Int64,
    spec::T3,
    x::T4)::Float64 where {T1,T4<:Real,T2<:UnivariateDistribution,T3<:MSetting}
    #
    dtf = dPower(NewDist(d, float(θ)), i)
    μ = mean(dtf)
    σ = std(dtf)

    specTemp = updatekL(dtf, spec)
    ψ((x^i - μ) / σ, specTemp)
end

function tfAVarU(θ::Vector{T1},
    d::T2,
    i::Int64,
    spec::T3,
    x::T4)::Float64 where {T1,T4<:Real,T2<:UnivariateDistribution,T3<:MSetting}
    #
    dtf = dPower(NewDist(d, float(θ)), i)
    μ = mean(dtf)
    σ = std(dtf)

    specTemp = updatekU(dtf, spec)
    ψ((x^i - μ) / σ, specTemp)
end

function tfAVarC(θ::Vector{T1},
    d::T2,
    i::Int64,
    spec::T3,
    x::T4)::Float64 where {T1,T4<:Real,T2<:UnivariateDistribution,T3<:MSetting}
    #
    dtf = dPower(NewDist(d, float(θ)), i)
    μ = mean(dtf)
    σ = std(dtf)

    c = findCorr(dtf, spec)
    ψ((x^i - μ) / σ, spec) - c
end

function tfAVarN(θ::Vector{T1},
    d::T2,
    i::Int64,
    spec::T3,
    x::T4)::Float64 where {T1,T4<:Real,T2<:UnivariateDistribution,T3<:MSetting}
    #
    dtf = dPower(NewDist(d, float(θ)), i)
    μ = mean(dtf)
    σ = std(dtf)

    ψ((x^i - μ) / σ, spec)
end

function Afun(θ::Vector{T1},
    d::T2,
    spec::Vector{T3},
    biasCorr::Union{Symbol,String}) where {T1<:Real,T2<:UnivariateDistribution,T3<:MSetting}
    #
    nPar = nParEff(d)
    outFun = function (xx)
        out = zeros(nPar, nPar)
        for i = 1:nPar
            out[i, :] = Calculus.gradient(vars -> eval(Expr(:call, Symbol("tfAVar", biasCorr), vars, d, i, spec[i], xx)), θ)
        end
        out
    end
    outFun
end

function Afun(θ::T1,
    d::T2,
    spec::T3,
    biasCorr::Union{Symbol,String}) where {T1<:Real,T2<:UnivariateDistribution,T3<:MSetting}
    #
    outFun = function (xx)
        Calculus.gradient(vars -> eval(Expr(:call, Symbol("tfAVar", biasCorr), vars, d, 1, spec, xx)), [θ])[1]
    end
    outFun
end

function Bfun(θ::Vector{T1},
    d::T2,
    spec::Vector{T3},
    biasCorr::Union{Symbol,String}) where {T1<:Real,T2<:UnivariateDistribution,T3<:MSetting}
    #
    nPar = nParEff(d)
    outFun = function (xx)
        temp = zeros(nPar)
        for i = 1:nPar
            temp[i] = eval(Expr(:call, Symbol("tfAVar", biasCorr), θ, d, i, spec[i], xx))
        end
        temp * temp'
    end
    outFun
end

function Bfun(θ::T1,
    d::T2,
    spec::T3,
    biasCorr::Union{Symbol,String}) where {T1<:Real,T2<:UnivariateDistribution,T3<:MSetting}
    #
    outFun = function (xx)
        eval(Expr(:call, Symbol("tfAVar", biasCorr), [θ], d, 1, spec, xx))^2
    end
    outFun
end

# Functions for asymptiotc variance

# If no sample given, Discrete, One parameter
"""
    AVar(d::UnivariateDistribution, spec::MSetting, biasCorr::Union{Symbol, String})
    AVar(d::UnivariateDistribution, spec::Vector{MSetting}, biasCorr::Union{Symbol, String})
    AVar(x::Vector{Real}, d::UnivariateDistribution, spec::MSetting, biasCorr::Union{Symbol, String})
    AVar(x::Vector{Real}, d::UnivariateDistribution, spec::Vector{MSetting}, biasCorr::Union{Symbol, String})

Asymptotic variance / covariance matrix of M-estimators.

If the fitted distribution `d` has multiple parameters, different specification used in estimation
can be provided.

The asymptotic variance / covariance matrix can be computed either for a fitted distribution `d` or 
estimated using the sample `c` and the fitted distribution `d`.

The argument `biasCorr` specifies which method was used in the estimation to correct for a potential bias.

If the fitted distribution `d` is continuous and no sample is provided, the computation relies on
numerical evaluation of vexpectations using the [Expectations.jl package](https://github.com/QuantEcon/Expectations.jl).
The precision of numerical integration is controlled by additional keyword argument `n` with default `n = 1000`.

# Example
```julia
d = Poisson(10)
x = rand(d, 200)
spec = Huber(1.5)
λ = Mfit(x, d, spec)
dFit = Poisson(λ)

AVar(dFit, spec, :L)
AVar(x, dFit, spec, :L)

d = Normal(0, 1)
x = rand(d, 200)
spec = [Huber(1.5), Huber(2.5)]
ests = Mfit(x, d, spec)
dFit = NewDist(d, ests)

AVar(dFit, spec, :L, n = 500)
AVar(x, dFit, spec, :L)
```

See [Stefanski and Book (2002)](https://www.jstor.org/stable/3087324) for theoretical background.
"""
function AVar(d::T1,
    spec::T2,
    biasCorr::Union{Symbol,String}=:L) where {T1<:Distribution{Univariate,Discrete},T2<:MSetting}
    #
    nPar = nParEff(d)
    if nPar > 1
        return AVar(d, fill(spec, nPar), biasCorr)
    end
    sup = ifelse(minimum(d) > -Inf, minimum(d), quantile(d, 0.001)):ifelse(maximum(d) < Inf, maximum(d), quantile(d, 0.999))
    prob = pdf.(d, sup)
    θ = getParams(d)
    A = 0.0
    B = 0.0
    fA = Afun(θ[1], d, spec, biasCorr)
    fB = Bfun(θ[1], d, spec, biasCorr)
    for i = 1:length(sup)
        A += fA(sup[i]) * prob[i]
        B += fB(sup[i]) * prob[i]
    end
    B / (A^2)
end

# If no sample given, Continuous, One parameter
function AVar(d::T1,
    spec::T2,
    biasCorr::Union{Symbol,String}=:L;
    n::Int64=1000) where {T1<:Distribution{Univariate,Continuous},T2<:MSetting}
    #
    nPar = nParEff(d)
    if nPar > 1
        return AVar(d, fill(spec, nPar), biasCorr)
    end
    if !isfinite(maximum(d) - minimum(d))
        E = expectation(truncated(d, quantile(d, 0.001), quantile(d, 0.999)), n=n)
    else
        E = expectation(d, n=n)
    end

    θ = getParams(d)
    A = 0.0
    B = 0.0
    fA = Afun(θ[1], d, spec, biasCorr)
    fB = Bfun(θ[1], d, spec, biasCorr)
    A = E(xx -> fA(xx))
    B = E(xx -> fB(xx))
    B / (A^2)
end


# If no sample given, Discrete, Multiple parameters
function AVar(d::T1,
    spec::Vector{T2},
    biasCorr::Union{Symbol,String}=:L) where {T1<:Distribution{Univariate,Discrete},T2<:MSetting}
    #
    nPar = nParEff(d)
    sup = ifelse(minimum(d) > -Inf, minimum(d), quantile(d, 0.001)):ifelse(maximum(d) < Inf, maximum(d), quantile(d, 0.999))
    prob = pdf.(d, sup)
    θ = getParams(d)

    A = zeros(nPar, nPar)
    B = zeros(nPar, nPar)

    fA = Afun(θ, d, spec, biasCorr)
    fB = Bfun(θ, d, spec, biasCorr)

    for i = 1:length(sup)
        A .+= fA(sup[i]) .* prob[i]
        B .+= fB(sup[i]) .* prob[i]
    end
    Ainv = inv(A)
    return Ainv * B * Ainv'
end


# If no sample given, Continuous, Multiple parameters
function AVar(d::T1,
    spec::Vector{T2},
    biasCorr::Union{Symbol,String}=:L;
    n::Int64=1000) where {T1<:Distribution{Univariate,Continuous},T2<:MSetting}
    #
    nPar = nParEff(d)
    if !isfinite(maximum(d) - minimum(d))
        E = expectation(truncated(d, quantile(d, 0.001), quantile(d, 0.999)), n=n)
    else
        E = expectation(d, n=n)
    end

    θ = getParams(d)
    A = zeros(nPar, nPar)
    B = zeros(nPar, nPar)
    fA = Afun(θ, d, spec, biasCorr)
    fB = Bfun(θ, d, spec, biasCorr)

    A = E(xx -> fA(xx))
    B = E(xx -> fB(xx))
    Ainv = inv(A)
    return Ainv * B * Ainv'
end

# If estimated from sample: d must be the fitted distribution(!)
# For single parameter
function AVar(x::Vector{T1},
    d::T2,
    spec::T3,
    biasCorr::Union{Symbol,String}=:L) where {T1<:Real,T2<:UnivariateDistribution,T3<:MSetting}
    #
    nPar = nParEff(d)
    if nPar > 1
        return AVar(x, d, fill(spec, nPar), biasCorr)
    end
    θ = getParams(d)

    A = 0.0
    B = 0.0
    fA = Afun(θ, d, spec, biasCorr)
    fB = Bfun(θ, d, spec, biasCorr)

    for i = x
        A += fA(i)
        B += fB(i)
    end
    n = length(x)
    return n * B / (A^2)
end

# If estimated from sample: d must be the fitted distribution(!)
# For multiple parameters
function AVar(x::Vector{T1},
    d::T2,
    spec::Vector{T3},
    biasCorr::Union{Symbol,String}=:L) where {T1<:Real,T2<:UnivariateDistribution,T3<:MSetting}
    #
    nPar = nParEff(d)
    θ = getParams(d)

    A = zeros(nPar, nPar)
    B = zeros(nPar, nPar)

    fA = Afun(θ, d, spec, biasCorr)
    fB = Bfun(θ, d, spec, biasCorr)

    for i = x
        A .+= fA(i)
        B .+= fB(i)
    end
    n = length(x)
    Ainv = inv(A)
    return (Ainv * B * Ainv') .* n
end