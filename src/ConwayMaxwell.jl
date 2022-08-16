# Implementation of the Conway-Maxwell-Poisson Distribution
# Mainly to check the fallback FInfo function

struct ConwayMaxwell{T <: Real} <: DiscreteUnivariateDistribution
    λ::T
    ν::T
    ConwayMaxwell{T}(λ::T, ν::T) where {T} = new{T}(λ, ν)
end

import Distributions.@check_args
function ConwayMaxwell(λ::T, ν::T; check_args=true) where {T <: Real}
    check_args && @check_args(ConwayMaxwell, (λ > zero(λ) && ν > zero(ν)) || (zero(λ) < λ < one(λ)) && ν >= zero(ν))
    return ConwayMaxwell{T}(λ, ν)
end

ConwayMaxwell(λ::Integer, ν::Integer) = ConwayMaxwell(float(λ), float(ν))
ConwayMaxwell(λ::Integer, ν::Float64) = ConwayMaxwell(float(λ), ν)
ConwayMaxwell(λ::Float64, ν::Integer) = ConwayMaxwell(λ, float(ν))

import Distributions.@distr_support
import Base.minimum, Base.maximum
@distr_support ConwayMaxwell 0 (d.λ == zero(typeof(d.λ)) ? 0 : Inf)

#### Conversions

import Base.convert
convert(::Type{ConwayMaxwell{T}}, λ::S, ν::S) where {T <: Real, S <: Real} = ConwayMaxwell(T(λ), T(ν))
convert(::Type{ConwayMaxwell{T}}, d::ConwayMaxwell{S}) where {T <: Real, S <: Real} = ConwayMaxwell(T(d.λ), T(d.ν), check_args=false)

### Parameters
import Distributions.params
params(d::ConwayMaxwell) = (d.λ, d.ν)
import Distributions.partype
partype(::ConwayMaxwell{T}) where {T} = T

function Z(λ::T1, ν::T2, atol::Float64 = 1e-10) where {T1 <: Number, T2 <:Real}
    out = one(λ)
    lλ = log(λ)
    δ = Inf
    j = 1
    while abs(δ) > atol
        δ = exp(j*lλ - ν*logabsgamma(j+1)[1])
        out += δ
        j += 1
        if j > 1000
            break
        end
    end

    return out
end

import Distributions.mean
function mean(d::ConwayMaxwell)
    (λ, ν) = params(d)
    out = λ
    lλ = log(λ)
    δ = Inf
    j = 2
    while δ > 1e-10
        δ = exp(j*lλ - ν*logabsgamma(j+1)[1])*j
        out += δ
        j += 1
        if j > 1000
            break
        end
    end
    return out/Z(λ, ν)
end

import Distributions.mode
function mode(d::ConwayMaxwell)
    floor(Int64, d.λ^(1/d.ν))
end

import Distributions.modes
function modes(d::ConwayMaxwell)
    (λ, ν) = params(d)
    λν = λ^(1/ν)
    isinteger(λν) ? [round(Int, λν) - 1, round(Int, λν)] : [floor(Int, λν)]
end

import Distributions.var
function var(d::ConwayMaxwell)
    (λ, ν) = params(d)
    out1 = λ
    out2 = λ
    lλ = log(λ)
    δ1 = Inf
    δ2 = Inf
    j = 2
    while δ2 > 1e-10
        temp = exp(j*lλ - ν*logabsgamma(j+1)[1])
        δ1 = temp*j
        δ2 = temp*j^2
        out1 += δ1
        out2 += δ2
        j += 1
        if j > 1000
            break
        end
    end
    z = Z(λ, ν)
    return out2/z - (out1/z)^2
end

import Distributions.skewness
function skewness(d::ConwayMaxwell)
    (λ, ν) = params(d)
    out1 = λ
    out2 = λ
    out3 = λ
    lλ = log(λ)
    δ1 = Inf
    δ2 = Inf
    δ3 = Inf
    j = 2
    while δ3 > 1e-10
        temp = exp(j*lλ - ν*logabsgamma(j+1)[1])
        δ1 = temp*j
        δ2 = temp*j^2
        δ3 = temp*j^3
        out1 += δ1
        out2 += δ2
        out3 += δ3
        j += 1
        if j > 1000
            break
        end
    end
    z = Z(λ, ν)
    Ex1 = out1/z
    Ex2 = out2/z
    Ex3 = out3/z

    s = Ex2 - Ex1^2
    return (Ex3 - 3*Ex1*s - Ex1^3)/(s^(3/2))
end

import Distributions.kurtosis
function kurtosis(d::ConwayMaxwell)
    (λ, ν) = params(d)
    out1 = λ
    out2 = λ
    out3 = λ
    out4 = λ
    lλ = log(λ)
    δ1 = Inf
    δ2 = Inf
    δ3 = Inf
    δ4 = Inf
    j = 2
    while δ4 > 1e-10
        temp = exp(j*lλ - ν*logabsgamma(j+1)[1])
        δ1 = temp*j
        δ2 = temp*j^2
        δ3 = temp*j^3
        δ4 = temp*j^4
        out1 += δ1
        out2 += δ2
        out3 += δ3
        out4 += δ4

        j += 1
        if j > 1000
            break
        end
    end
    z = Z(λ, ν)
    Ex1 = out1/z
    Ex2 = out2/z
    Ex3 = out3/z
    Ex4 = out4/z
    s = Ex2 - Ex1^2
    (Ex4 - 4*Ex3*Ex1 + 6*s*Ex1^2 + 3*Ex1^4)/s^2
end

import Distributions.mgf
function mgf(d::ConwayMaxwell, t::Real)
    (λ, ν) = params(d)
    return Z(exp(t*λ), ν)/Z(λ, ν)
end

import Distributions.cf
function cf(d::ConwayMaxwell, t::Real)
    (λ, ν) = params(d)
    return Z(λ * (cis(t) - 1), ν)/Z(λ, ν)
end

import Distributions.pdf
function pdf(d::ConwayMaxwell, x::Integer)
    if x < 0
        return 0.0
    end
    (λ, ν) = params(d)

    exp(x*log(λ) - ν*logabsgamma(x+1)[1])/Z(λ, ν)
end

function pdf(d::ConwayMaxwell, x::Real)
    round(Int, x) == x ? pdf(d, round(Int, x)) : 0.0
end

import Distributions.cdf
function cdf(d::ConwayMaxwell, x::Integer)
    if x < 0
        return 0.0
    end
    (λ, ν) = params(d)
    lλ = log(λ)

    out = 1.0
    for j = 1:x
        out += exp(j*lλ - ν*logabsgamma(j+1)[1])
    end

    out/Z(λ, ν)
end

function cdf(d::ConwayMaxwell, x::Real)
    if x < 0
        return 0.0
    end
    cdf(d, floor(Int, x))
end


# Random Number Generation

# Rejection Sampling
# For ν > 1 -> Poisson distribution
# For ν < 1 -> Geometric distribution
# See
# Beson, A and Friel, N. (2021)
# "Bayesian Inference, Model Selection and Likelihood Estimation using Fast
# Rejection Sampling: The Conway-Maxwell-Poisson Distribution"
# in Bayesian Analysis 16 Nr. 3 pp 905-931

# Functions to compute the fraction of densities efficiently
# Problem: Repeated computation of normalizing constants and
#          accolations of pdf functions for Geometric/Poisson
function densFracP(x::Int64,
                   λ::Float64,
                   ν::Float64,
                   λp::Float64,
                   c::Float64)::Float64
    exp(x*log(λp/λ) + (ν - 1)*logabsgamma(x+1)[1])*c
end

function densFracG(x::Int64,
                   λ::Float64,
                   ν::Float64,
                   p::Float64,
                   c::Float64)::Float64
    exp(x*log((1-p)/λ) + ν*logabsgamma(x+1)[1])*c
end

import Distributions.rand
function rand(d::ConwayMaxwell)
    (λ, ν) = params(d)
    if ν == 1
        return rand(Poisson(λ))
    end

    out = zero(Int64)
    z = Z(λ, ν)
    μ = mean(d)

    if ν < 1
        p = 2*ν/(2*μ*ν + 1 + ν)
        dg = Geometric(p)
        ym = floor(Int64, μ/((1 - p)^(1/ν)))
        c = p*z
        scl = c/densFracG(ym, λ, ν, p, c)
        while true
            cand = rand(dg)
            α = 1/densFracG(cand, λ, ν, p, scl)
            if rand() <= α
                return cand
            end
        end
    end

    if ν > 1
        λp = ceil(μ)
        dp = Poisson(λp)
        c = exp(-λp)*z
        scl = c/densFracP(Int64(λp), λ, ν, λp, c)

        while true
            cand = rand(dp)
            α = 1/densFracP(cand, λ, ν, λp, scl)
            if rand() <= α
                return cand
            end
        end
    end
end

function rand(d::ConwayMaxwell, n::Int64)
    (λ, ν) = params(d)
    if ν == 1
        return rand(Poisson(λ), n)
    end

    z = Z(λ, ν)
    out = zeros(Int64, n)
    μ = mean(d)

    if ν < 1
        p = 2*ν/(2*μ*ν + 1 + ν)
        dg = Geometric(p)
        ym = floor(Int64, μ/((1 - p)^(1/ν)))
        c = p*z
        scl = c/densFracG(ym, λ, ν, p, c)
        i = 1
        while true
            cand = rand(dg)
            α = 1/densFracG(cand, λ, ν, p, scl)
            if rand() <= α
                out[i] = cand
                if i >= n
                    return out
                else
                    i += 1
                end
            end
        end
    end

    if ν > 1
        λp = ceil(μ)
        dp = Poisson(λp)
        c = exp(-λp)*z
        scl = c/densFracP(Int64(λp), λ, ν, λp, c)
        i = 1
        while true
            cand = rand(dp)
            α = 1/densFracP(cand, λ, ν, λp, scl)
            if rand() <= α
                out[i] = cand
                if i >= n
                    return out
                else
                    i += 1
                end
            end
        end
    end
end

function rand(d::ConwayMaxwell, dims::Dims)
    x = rand(d, prod(dims))
    reshape(x, dims)
end

function rand(d::ConwayMaxwell, dim1::Int, moredims::Int...)
    x = rand(d, dim1*prod(moredims))
    reshape(x, (dim1, moredims...))
end
