# Struct for powers of distributions
"""
    dPower(d::UnivariateDistribution, p::Int64)

A data type describing the power of a univariate random variable X.
Given the distribution of X: `d` and an integer `i`, the type describes ``X^i``.

# Example
```julia
    dX = Poisson(10)
    dY = dPower(d, 2)
```
"""
struct dPower
    d::UnivariateDistribution
    p::Int64
end

# Short version of constructor
### Removed due to type piracy ###
# import Base.^
# ^(d::T, p::Int64) where {T <: Distribution} = begin
#     dPower(d, p)
# end::dPower
#
# ^(d::dPower, p::Int64) = begin
#     dPower(d.d, d.p*p)
# end

# Length and iterate needed for vector-wise evaluation of pdf/cdf/...
import Base.length
function length(x::dPower)::Int64
    1
end

import Base.iterate
function iterate(x::dPower)
    (x, nothing)
end
function iterate(x::dPower, foo::Nothing)

end

# Random number generation
import Base.rand
"""
    rand(d::dPower)
    rand(d::dPower, n::Int64)
    rand(d::dPower, dims::Dims)

Random number generation for data type `dPower`.

# Example
```julia
dX = Poisson(10)
dY = dPower(dX, 2)
rand(dY)
rand(dY, 100)
rand(dY, (100, 100))
```

See also [`dPower`](@ref dPower).
"""
function rand(d::dPower)::Float64
    rand(d.d)^d.p
end

function rand(d::dPower, n::Int64)::Vector{Float64}
    rand(d.d, n) .^ d.p
end

function rand(d::dPower, dims::Dims)
    rand(d.d, dims) .^ d.p
end

# Minimum and maximum of distributions support
import Distributions.minimum
"""
    minimum(d::dPower)

Minimum for data type `dPower`.

# Example
```julia
dX = Normal(5, 1)
dY = dPower(dX, 2)
minimum(dPower)
```

See also [`dPower`](@ref dPower).
"""
function minimum(d::dPower)::Float64
    m = minimum(d.d)
    M = maximum(d.d)
    x1 = m^d.p
    x2 = M^d.p
    if (sign(m) != sign(M)) & iseven(d.p)
        return 0.0
    end

    return minimum([x1, x2])
end

import Distributions.maximum
"""
    maximum(d::dPower)

Maximum for data type `dPower`.

# Example
```julia
dX = Normal(5, 1)
dY = dPower(dX, 2)
maximum(dPower)
```

See also [`dPower`](@ref dPower).
"""
function maximum(d::dPower)::Float64
    x1 = minimum(d.d)^d.p
    x2 = maximum(d.d)^d.p
    return maximum([x1, x2])
end

# Check if x is in support of distribution
import Distributions.insupport
"""
    insupport(d::dPower, x::Real)

Check if `x` is in the support of distribution `d`, where `d` is of type `dPower`.

# Example
```julia
dX = Poisson(10)
dY = dPower(dX, 2)
insupport(dPower, 9)
```

See also [`dPower`](@ref dPower).
"""
function insupport(d::dPower, x::T)::Bool where {T<:Real}
    if !(minimum(d) <= x <= maximum(d))
        return false
    end

    x1 = abs(x)^(1 / d.p)
    if abs(x1 - Int64(round(x1))) <= 1e-05
        x1 = Int64(round(x1))
    end
    x2 = -abs(x)^(1 / d.p)
    if abs(x2 - Int64(round(x2))) <= 1e-05
        x2 = Int64(round(x2))
    end

    if insupport(d.d, x1) | insupport(d.d, x2)
        return true
    else
        return false
    end
end

# Function to order support of discrete distributions for even powers
# Only internal
# Might not be needed?
function iterateSupport(d::dPower, iMax::Int64)::Vector{Float64}
    outRaw = zeros(Int64, 2 * iMax)
    outRaw[1] = Int64(round(mean(d.d)))
    goup = outRaw[1] <= mean(d.d)
    Min = minimum(d.d)
    Max = maximum(d.d)
    currMin = outRaw[1] - 1
    currMax = outRaw[1] + 1

    for i = 2:(2*iMax)
        if currMin < Min
            goup = true
        elseif currMax > Max
            goup = false
        else
            goup = !goup
        end

        if goup
            outRaw[i] = currMax
            currMax += 1
        else
            outRaw[i] = currMin
            currMin -= 1
        end
    end

    return outRaw[1:iMax]
end

# Cumulative distribution function
import Distributions.cdf
"""
    cdf(d::dPower, x::Real)

Cumulative distribution function for `dPower` distribution.

# Example
```julia
dX = Normal()
dY = dPower(dX, 2)
cdf(dPower, 2)
```

See also [`dPower`](@ref dPower).
"""
function cdf(d::dPower, x::T)::Float64 where {T<:Real}
    if iseven(d.p)
        if (x < minimum([minimum(d)^d.p, 0]))
            return 0.0
        end
        if (x > maximum(d))
            return 1.0
        end

        if supertype(typeof(d.d)) == Distribution{Univariate,Continuous}
            x1 = abs(x)^(1 / d.p)
            x2 = -abs(x)^(1 / d.p)
            return cdf(d.d, x1) - cdf(d.d, x2)
        elseif supertype(typeof(d.d)) == Distribution{Univariate,Discrete}
            sup = iterateSupport(d, 10)
            while true
                if any(sup .^ d.p .> x)
                    sup = sup[sup .^ d.p.<=x]
                    break
                else
                    sup = iterateSupport(d, 2 * length(sup))
                end
            end
            if length(sup) == 0
                return 0
            end
            return sum(pdf.(d.d, sup))
        end
    else
        x1 = round(sign(x) * abs(x)^(1 / d.p), digits = 10)
        return cdf(d.d, x1)
    end
end

# Probability density function
# If p is even, fold distribution otherwise density transformation
import Distributions.pdf
"""
    pdf(d::dPower, x::Real)

Probability (density) function for `dPower` distribution.

# Example
```julia
dX = Normal()
dY = dPower(dX, 2)
pdf(dPower, 2)
```

See also [`dPower`](@ref dPower).
"""
function pdf(d::dPower, x::T)::Float64 where {T<:Real}
    if supertype(typeof(d.d)) == Distribution{Univariate,Discrete}
        if iseven(d.p)
            if x < 0
                return 0.0
            end
            x1 = round(abs(x)^(1 / d.p), digits = 10)
            x2 = round(-abs(x)^(1 / d.p), digits = 10)
            return sum(pdf.(d.d, unique([x1, x2])))
        else
            x1 = round(sign(x) * abs(x)^(1 / d.p), digits = 10)
            return pdf(d.d, x1)
        end
    elseif supertype(typeof(d.d)) == Distribution{Univariate,Continuous}
        if iseven(d.p)
            if x < 0
                return 0
            end
            x1 = abs(x)^(1 / d.p)
            x2 = -abs(x)^(1 / d.p)
            return (pdf(d.d, x1) + pdf(d.d, x2)) * abs(x)^(1 / d.p - 1) / d.p
        else
            x1 = round(abs(x) .^ (1 / d.p), digits = 10)
            return pdf(d.d, sign(x) * x1) * abs(x)^(1 / d.p - 1) / d.p
        end
    else
        error("Invalid distribution")
    end
end

import Distributions.logpdf
"""
    logpdf(d::dPower, x::Real)

Logarithm of probability (density) function for `dPower` distribution.

# Example
```julia
dX = Normal()
dY = dPower(dX, 2)
logpdf(dPower, 2)
```

See also [`dPower`](@ref dPower).
"""
function logpdf(d::dPower, x::T)::Float64 where {T<:Real}
    log(pdf(d, x))
end

# Quantile function
# For even p and continuous distribution, use bisection search
# with 10 digit accuracy
import Distributions.quantile
"""
    quantile(d::dPower, q::Real)

`q`-Quantile of a `dPower` random variable.

# Example
```julia
dX = Normal()
dY = dPower(dX, 2)
quantile(dPower, 0.95)
```

See also [`dPower`](@ref dPower).
"""
function quantile(d::dPower, q::T)::Float64 where {T<:Real}
    if iseven(d.p)
        if minimum(d.d) >= 0
            return quantile(d.d, q)^d.p
        end
        if supertype(typeof(d.d)) == Distribution{Univariate,Continuous}
            lower = 0
            Fl = cdf(d, lower)
            upper = 10
            Fu = cdf(d, upper)
            while true
                if sign(Fl - q) == sign(Fu - q)
                    lower = copy(upper)
                    Fl = copy(Fu)
                    upper = upper * 2
                    Fu = cdf(d, upper)
                else
                    break
                end
            end

            while true
                cand = (upper + lower) / 2
                Fcand = cdf(d, cand)
                if Fcand <= q
                    lower = copy(cand)
                    Fl = copy(Fcand)
                else
                    upper = copy(cand)
                    Fu = copy(Fcand)
                end
                if abs(upper - lower) <= 1e-10
                    return upper
                end
            end
        elseif supertype(typeof(d.d)) == Distribution{Univariate,Discrete}
            sup = iterateSupport(d, 10)
            while true
                if cdf(d, sup[end]^d.p) >= q
                    break
                else
                    sup = iterateSupport(d, 2 * length(sup))
                end
            end
            cs = cdf.(d, sup .^ d.p)
            if all(cs .> q)
                return 0
            else
                return minimum(sup[cs.>=q] .^ d.p)
            end
        end
    else
        return quantile(d.d, q)^d.p
    end
end

# Mean of distribution
# For powers up to p = 4 use skewness/kurtosis functions
# For higher powers, use Expectations.jl -> Numerical integration

import Distributions.skewness
"""
    skewness(d::dPower)

Skewness of a `dPower` random variable.

# Example
```julia
dX = Normal()
dY = dPower(dX, 2)
skewness(dPower)
```

See also [`dPower`](@ref dPower).
"""
function skewness(d::dPower)
    m = mean(d)
    v = var(d)
    (mean(dPower(d.d, d.p*3)) + 2*m*v - m^3)/(v^(3/2))
end

function skewness(d::T) where {T<:Distribution{Univariate,Continuous}}
    if isfinite(maximum(d) - minimum(d))
        E = expectation(d, n=1000)
    else
        E = expectation(truncated(d, quantile(d, 0.0001), quantile(d, 0.9999)), n=1000)
    end
    μ = mean(d)
    σ = std(d)
    return E(x -> ((x - μ) / σ)^3)
end

function skewness(d::NoncentralF{Float64})::Float64
    λ = d.λ
    ν1 = d.ν1
    ν2 = d.ν2
    temp = (ν1 + ν2 - 2)
    centr = 8 * ν2^3 * (2 * λ^3 + 6 * temp * λ^2 + 3 * temp * (2 * ν1 + ν2 + 2) * λ + ν1 * temp * (2 * ν1 + ν2 - 2))
    centr = centr / (ν1^3 * (ν2 - 2)^3 * (ν2 - 4) * (ν2 - 6))
    σ = std(d)
    return centr / σ^3
end

import Distributions.kurtosis
"""
    kurtosis(d::dPower)

Kurtosis of a `dPower` random variable.

# Example
```julia
dX = Normal()
dY = dPower(dX, 2)
kurtosis(dPower)
```

See also [`dPower`](@ref dPower).
"""
function kurtosis(d::dPower)
    m1 = mean(d)
    m2 = mean(dPower(d.d, d.p*2))
    m3 = mean(dPower(d.d, d.p*3))
    m4 = mean(dPower(d.d, d.p*4))

    m4 - 4*m3*m1 + 6*m2*m1^2 - 3*m1^4
end

function kurtosis(d::T) where {T<:Distribution{Univariate,Continuous}}
    if isfinite(maximum(d) - minimum(d))
        E = expectation(d, n=1000)
    else
        E = expectation(truncated(d, quantile(d, 0.0001), quantile(d, 0.9999)), n=1000)
    end
    μ = mean(d)
    σ = std(d)
    return E(x -> ((x - μ) / σ)^4) - 3
end

# Correct typo in Distribution.jl function for NoncentralF variance
import Distributions.var
function var(d::NoncentralF{Float64})
    2 * d.ν2^2 * ((d.ν1 + d.λ)^2 + (d.ν2 - 2) * (d.ν1 + 2d.λ)) / (d.ν1^2 * (d.ν2 - 2)^2 * (d.ν2 - 4))
end

import Distributions.mean
"""
    mean(d::dPower)

Mean/expectation of a `dPower` random variable.

# Example
```julia
dX = Normal()
dY = dPower(dX, 2)
mean(dPower)
```

See also [`dPower`](@ref dPower).
"""
function mean(d::dPower)
    if d.p == 1
        return mean(d.d)
    elseif d.p == 2
        return var(d.d) + mean(d.d)^2
    elseif d.p == 3
        γ = skewness(d.d)
        σ = std(d.d)
        μ = mean(d.d)
        Ex2 = σ^2 + μ^2
        return γ * σ^3 + 3 * Ex2 * μ - 2 * μ^3
    elseif d.p == 4
        κ = kurtosis(d.d, false)
        γ = skewness(d.d)
        σ = std(d.d)
        μ = mean(d.d)
        Ex2 = σ^2 + μ^2
        Ex3 = γ * σ^3 + 3 * Ex2 * μ - 2 * μ^3
        return κ * σ^4 + 4 * Ex3 * μ - 6 * Ex2 * μ^2 + 3 * μ^4
    else
        p = d.p
        if supertype(typeof(d.d)) == Distribution{Univariate,Continuous}
            if isfinite(minimum(d.d) - maximum(d.d))
                E = expectation(d.d, n=1000)
            else
                E = expectation(truncated(d.d, quantile(d.d, 0.0001), quantile(d.d, 0.9999)), n=1000)
            end
            return E(x -> x^p)
        else
            sup = collect(quantile(d.d, 0.0001):quantile(d.d, 0.9999))
            pmf = pdf.(d.d, sup)
            vals = sup .^ p
            return vals'pmf
        end
    end
end

# Fallback functions in case only pdf and min/max known
function mean(d::T) where {T<:Distribution{Univariate,Continuous}}
    lb = quantile(d, 0.0001)
    ub = quantile(d, 0.9999)
    quadgk(x -> x * pdf(d, x), lb, ub)[1]
end

function mean(d::T) where {T<:Distribution{Univariate,Discrete}}
    lb = quantile(d, 0.0001)
    ub = quantile(d, 0.9999)
    sup = lb:ub
    sup'pdf.(d, sup)
end

# Variance function for powers, uses mean function
import Distributions.var
"""
    var(d::dPower)

Variance of a `dPower` random variable.

# Example
```julia
dX = Normal()
dY = dPower(dX, 2)
var(dPower)
```

See also [`dPower`](@ref dPower).
"""
function var(d::dPower)
    mean(dPower(d.d, 2 * d.p)) - mean(d)^2
end

import Distributions.std
"""
    std(d::dPower)

Standard deviation of a `dPower` random variable.

# Example
```julia
dX = Normal()
dY = dPower(dX, 2)
std(dPower)
```

See also [`dPower`](@ref dPower).
"""
function std(d::dPower)
    sqrt(var(d))
end

# Fallback functions in case only pdf and min/max known
function var(d::T) where {T<:Distribution{Univariate,Continuous}}
    lb = quantile(d, 0.0001)
    ub = quantile(d, 0.9999)
    quadgk(x -> x^2 * pdf(d, x), lb, ub)[1] - mean(d)^2
end

function var(d::T) where {T<:Distribution{Univariate,Discrete}}
    lb = quantile(d, 0.0001)
    ub = quantile(d, 0.9999)
    sup = lb:ub
    (sup .^ 2)'pdf.(d, sup) - mean(d)^2
end

# CDF fallback function if only pdf known
import Distributions.cdf
function cdf(d::T1, x::T2)::Float64 where {T1<:Distribution{Univariate,Continuous},T2<:Real}
    lb = minimum(d)
    quadgk(z -> pdf(d, z), lb, x)[1]
end

function cdf(d::T1, x::T2)::Float64 where {T1<:Distribution{Univariate,Discrete},T2<:Real}
    lb = minimum(d)
    if !isfinite(lb)
        lb = mean(d) - 10 * std(d)
    end
    xx = Int64(floor(x))
    sum((z -> pdf(d, z)).(lb:xx))
end

# Quantile fallback function
# CDF must be known or alternatively derived from pdf
import Distributions.quantile
function quantile(d::T1, q::T2)::Float64 where {T1<:Distribution{Univariate,Continuous},T2<:Real}
    if !(0 <= q <= 1)
        error("Invalid q")
    end

    σ = std(d)

    lb = mean(d) - 3 * σ
    ub = mean(d) + 3 * σ

    while true
        Fl = cdf(d, lb)
        Fu = cdf(d, ub)
        if sign(Fl - q) == sign(Fu - q)
            lb = lb - 3 * σ
            ub = ub + 3 * σ
        else
            break
        end
    end

    find_zero(z -> cdf(d, z) - q, (lb, ub), Bisection(), atol=1e-05)
end

function quantile(d::T1, q::T2)::Float64 where {T1<:Distribution{Univariate,Discrete},T2<:Real}
    if !(0 <= q <= 1)
        error("Invalid q")
    end

    σ = std(d)

    lb = floor(Int, maximum([minimum(d), mean(d) - 3 * σ]))
    ub = ceil(Int, minimum([maximum(d), mean(d) + 3 * σ]))

    while cdf(d, ub) <= q
        ub += ceil(Int64, 3 * std(d))
    end

    while cdf(d, lb) >= q
        lb -= floor(Int64, 3 * std(d))
    end

    sup = lb:ub
    p = cdf.(d, sup)

    minimum(sup[p.>=q])
end


# Function to create distribution with new parameters
"""
    NewDist(d::UnivariateDistribution, θ::Vector{Real})
    
Create a new distribution object with updated parameters θ.
Parameters not be estimated are ignored.

# Example
```julia
d1 = Normal()
NewDist(d1, [0, 1])

d2 = Binomial(10, 0.4)
NewDist(d2, [0.7])
```
"""
function NewDist(d::T1, θ::Vector{T2})::T1 where {T1<:UnivariateDistribution,T2<:Real}
    nPar = length(θ)
    if nPar == 1
        return typeof(d)(θ[1])
    elseif nPar == 2
        return typeof(d)(θ[1], θ[2])
    elseif nPar == 3
        return typeof(d)(θ[1], θ[2], θ[3])
    elseif nPar == 4
        return typeof(d)(θ[1], θ[2], θ[3], θ[4])
    else
        error("More than 4 parameters not yet supported")
    end
end
# There must be a better way...

function NewDist(d::UnivariateDistribution, θ::T) where {T <: Real}
    NewDist(d, [θ])
end

# Exceptions for distributions with fixed parameters
function NewDist(d::BetaBinomial, θ::Vector{T}) where {T<:Real}
    BetaBinomial(d.n, θ[1], θ[2])
end

function NewDist(d::Binomial, θ::Vector{T}) where {T<:Real}
    Binomial(d.n, θ[1])
end

function NewDist(d::Erlang, θ::Vector{T}) where {T<:Real}
    Erlang(d.α, θ[1])
end

function NewDist(d::NormalCanon, θ::Vector{T}) where {T<:Real}
    NormalCanon(θ[1], θ[2])
end

function NewDist(d::Pareto, θ::Vector{T}) where {T<:Real}
    Pareto(d.α, θ[1])
end

function NewDist(d::VonMises, θ::Vector{T}) where {T<:Real}
    VonMises(θ[1], θ[2])
end

function NewDist(d::GeneralizedTDist, θ::Vector{T}) where {T<:Real}
    GeneralizedTDist(θ[1], θ[2], d.ν)
end
