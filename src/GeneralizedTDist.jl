# Implementation of the shifted T Distribution

"""
    GeneralizedTDist(μ::Real, σ::Real,  ν::Real)
    
The T distribution with parameters `μ`, `σ` and `ν`

# Example
```julia
d = GeneralizedTDist(1, 2, 10)
mean(d)
pdf(d, 1)
```
"""
struct GeneralizedTDist{T<:Real} <: Distribution{Univariate,Continuous}
    μ::T
    σ::T
    ν::T
    GeneralizedTDist{T}(μ::T, σ::T, ν::T) where {T} = new{T}(μ, σ, ν)
end

import Distributions.@check_args
function GeneralizedTDist(μ::T, σ::T, ν::T; check_args=true) where {T<:Real}
    check_args && @check_args(GeneralizedTDist, σ > zero(σ) && ν > zero(ν))
    return GeneralizedTDist{T}(μ, σ, ν)
end

GeneralizedTDist(μ::Real, σ::Real, ν::Real) = GeneralizedTDist(promote(μ, σ, ν)...)
GeneralizedTDist(μ::Integer, σ::Integer, ν::Integer) = GeneralizedTDist(float(μ), float(σ), float(ν))

import Distributions.@distr_support
@distr_support GeneralizedTDist -Inf Inf

function minimum(d::GeneralizedTDist)
    return -Inf
end
function maximum(d::GeneralizedTDist)
    return Inf
end

#### Conversions
import Base.convert
convert(::Type{GeneralizedTDist{T}}, μ::S, σ::S, ν::S) where {T<:Real,S<:Real} = GeneralizedTDist(T(μ), T(σ), T(ν))
convert(::Type{GeneralizedTDist{T}}, d::GeneralizedTDist{S}) where {T<:Real,S<:Real} = GeneralizedTDist(T(d.μ), T(d.σ), T(d.ν), check_args=false)

#### Parameters
import Distributions.params
params(d::GeneralizedTDist) = (d.μ, d.σ, d.ν)
import Distributions.partype
@inline partype(d::GeneralizedTDist{T}) where {T<:Real} = T

#### Statistics

import Distributions.mean
mean(d::GeneralizedTDist{T}) where {T<:Real} = ((μ, σ, ν) = params(d);
ν > 1 ? μ : T(NaN))

import Distributions.mode
function mode(d::GeneralizedTDist)
    (μ, σ, ν) = params(d)
    return μ
end

import Distributions.modes
modes(d::GeneralizedTDist) = [mode(d)]

import Distributions.var
function var(d::GeneralizedTDist{T}) where {T<:Real}
    (μ, σ, ν) = params(d)
    if ν <= 1
        return T(NaN)
    elseif ν <= 2
        return Inf
    end

    σ^2 * ν / (ν - 2)
end

import Distributions.skewness
function skewness(d::GeneralizedTDist{T}) where {T<:Real}
    (μ, σ, ν) = params(d)
    if ν > 3
        return 0
    else
        return T(NaN)
    end
end

import Distributions.kurtosis
function kurtosis(d::TDist{T}) where {T<:Real}
    (μ, σ, ν) = params(d)
    if ν <= 2
        return T(NaN)
    elseif ν <= 4
        return Inf
    end

    6 / (ν - 4)
end

#### Evaluation

import Distributions.pdf
function pdf(d::GeneralizedTDist, x::Real)::Float64
    pdf(TDist(d.ν), (x - d.μ) / d.σ) / d.σ
end

import Distributions.logpdf
function logpdf(d::GeneralizedTDist, x::Real)::Float64
    log(pdf(d, x))
end

import Distributions.cdf
function cdf(d::GeneralizedTDist, x::Real)::Float64
    cdf(TDist(d.ν), (x - d.μ) / d.σ)
end

import Distributions.quantile
function quantile(d::GeneralizedTDist, p::Real)::Float64
    d.σ * quantile(TDist(d.ν), p) + d.μ
end

import Base.length
function length(x::GeneralizedTDist)::Int64
    1
end

import Base.iterate
function iterate(x::GeneralizedTDist)
    (x, nothing)
end
function iterate(x::GeneralizedTDist, foo::Nothing)

end

import Distributions.rand
function rand(d::GeneralizedTDist)
    d.σ * rand(TDist(d.ν)) + d.μ
end

function rand(d::GeneralizedTDist, n::Integer)
    d.σ .* rand(TDist(d.ν), n) .+ d.μ
end

function rand(d::GeneralizedTDist, dims::Dims)
    d.σ .* rand(TDist(d.ν), dims) .+ d.μ
end

function rand(d::GeneralizedTDist, dim1::Int, moredims::Int...)
    d.σ .* rand(TDist(d.ν), (dim1, moredims...)) .+ d.μ
end

function checkParam(d::GeneralizedTDist, θ::Vector{T}) where {T<:Real}
    (θ[2] > 0) ? true : false
end