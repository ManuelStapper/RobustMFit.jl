# Implementation of the Dagum Distribution
# Mainly to check the fallback FInfo function

"""
    Dagum(a::Real, b::Real, p::Real)
    
The Dagum distribution with parameters `a`, `b` and `p`, see [Wikipedia](https://en.wikipedia.org/wiki/Dagum_distribution)

# Example
```julia
d = Dagum(2, 1, 1)
mean(d)
pdf(d, 1)
```
"""
struct Dagum{T<:Real} <: Distribution{Univariate,Continuous}
    a::T
    b::T
    p::T
    Dagum{T}(a::T, b::T, p::T) where {T} = new{T}(a, b, p)
end

import Distributions.@check_args
function Dagum(a::T, b::T, p::T; check_args=true) where {T<:Real}
    check_args && @check_args(Dagum, a > zero(a) && b > zero(b) && p > zero(p))
    return Dagum{T}(a, b, p)
end

Dagum(a::Real, b::Real, p::Real) = Dagum(promote(a, b, p)...)
Dagum(a::Integer, b::Integer, p::Integer) = Dagum(float(a), float(b), float(p))

import Distributions.@distr_support
@distr_support Dagum 0.0 Inf

#### Conversions
import Base.convert
convert(::Type{Dagum{T}}, a::S, b::S, p::S) where {T<:Real,S<:Real} = Dagum(T(a), T(b), T(p))
convert(::Type{Dagum{T}}, d::Dagum{S}) where {T<:Real,S<:Real} = Dagum(T(d.a), T(d.b), T(d.p), check_args=false)

#### Parameters
import Distributions.params
params(d::Dagum) = (d.a, d.b, d.p)
import Distributions.partype
@inline partype(d::Dagum{T}) where {T<:Real} = T

#### Statistics

import Distributions.mean
mean(d::Dagum{T}) where {T<:Real} = ((a, b, p) = params(d); a > 1 ? -b / a * (gamma(-1 / a) * gamma(1 / a + p)) / gamma(p) : T(NaN))

import Distributions.mode
function mode(d::Dagum)
    (a, b, p) = params(d)
    return b * ((a * p - 1) / (a + 1))^(-1 / a)
end

import Distributions.modes
modes(d::Dagum) = [mode(d)]

import Distributions.var
function var(d::Dagum{T}) where {T<:Real}
    (a, b, p) = params(d)
    if a <= 2
        return T(NaN)
    end

    -b^2 / a^2 * (2 * a * gamma(-2 / a) * gamma(2 / a + p) / gamma(p) + (gamma(-1 / a) * gamma(1 / a + p) / gamma(p))^2)
end

import Distributions.skewness
function skewness(d::Dagum)
    (a, b, p) = params(d)
    μ = mean(d)
    s = var(d)
    Ex3 = b^3 * beta(p + 3 / a, 1 - 3 / a) / beta(p, 1)
    (Ex3 - 3 * μ * s - μ^3) / (s^(3 / 2))
end

import Distributions.kurtosis
function kurtosis(d::Dagum)
    (a, b, p) = params(d)
    μ = mean(d)
    s = var(d)
    Ex3 = b^3 * beta(p + 3 / a, 1 - 3 / a) / beta(p, 1)
    Ex4 = b^4 * beta(p + 4 / a, 1 - 4 / a) / beta(p, 1)
    (Ex4 - 4 * Ex3 * μ + 6 * s * μ^2 + 3 * μ^4) / s^2 - 3
end

#### Evaluation

import Distributions.pdf
function pdf(d::Dagum, x::Real)::Float64
    (a, b, p) = params(d)
    x <= 0 ? 0.0 : a * p / x * ((x / b)^(a * p)) / (((x / b)^a + 1)^(p + 1))
end

import Distributions.cdf
function cdf(d::Dagum, x::Real)::Float64
    (a, b, p) = params(d)
    x <= 0 ? 0.0 : (1 + (x / p) .^ (-a))^(-p)
end

import Distributions.quantile
function quantile(d::Dagum, p::Real)::Float64
    d.b * (p .^ (-1 / d.p) - 1)^(-1 / d.a)
end

import Base.length
function length(x::Dagum)::Int64
    1
end

import Base.iterate
function iterate(x::Dagum)
    (x, nothing)
end
function iterate(x::Dagum, foo::Nothing)

end

import Distributions.rand
function rand(d::Dagum)
    (a, b, p) = params(d)
    b * (rand(Chisq(2 * p)) / rand(Chisq(2)))^(1 / a)
end

function rand(d::Dagum, n::Integer)
    (a, b, p) = params(d)
    c1 = rand(Chisq(2 * p), n)
    c2 = rand(Chisq(2), n)
    b .* (c1 ./ c2) .^ (1 / a)
end

function rand(d::Dagum, dims::Dims)
    (a, b, p) = params(d)
    c1 = rand(Chisq(2 * p), dims)
    c2 = rand(Chisq(2), dims)
    b .* (c1 ./ c2) .^ (1 / a)
end

function rand(d::Dagum, dim1::Int, moredims::Int...)
    (a, b, p) = params(d)
    c1 = rand(Chisq(2 * p), (dim1, moredims...))
    c2 = rand(Chisq(2), (dim1, moredims...))
    b .* (c1 ./ c2) .^ (1 / a)
end


# Other functions
function checkParam(d::Dagum, θ::Vector{T}) where {T<:Real}
    all(θ .> 0) ? true : false
end
