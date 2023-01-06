# Functions to compute gradient of Moments-To-Parameter functions
# out[i, j]: derivative parameter j with respect to μ_i

### General functions
"""
    MTPder(μ::Vector{Real}, d::UnivariateDistribution)

Derivative of the mapping from moments to parameters of a unviariate distribution.
`μ` must be a vector of the first p raw moments of the distribution, given that p
is the number of parameters to be estimated.
Parameters of the distribution `d` are ignored.

Hardcoded in closed form for 25 commonly used distributions.
For other distributions relies on the numerical gradient.

Output is a matrix where i-th row and j-th column gives the derivative of j-th parameter w.r.t the i-th moment.

# Example
```julia
d1 = Normal()
MTP([1, 5], d1)
d2 = Poisson(10)
MTP([5], d2)
```

See also [`nParEff`](@ref nParEff) for the effective number of parameters and
[`MTP`](@ref MTP) for the mapping from moments to parameters. 
"""
function MTPder(μ::Vector{T1}, d::T2) where {T1<:Real,T2<:UnivariateDistribution}
    nPar = length(μ)
    out = zeros(nPar, nPar)
    for j = 1:nPar
        out[:, j] = Calculus.gradient(x -> MTP(x, d)[j])(μ)
    end
    return out
end

# Distribution-free function
function MTPder(μ::Vector{T1}) where {T1<:Real}
    n = length(μ)

    if n > 4
        error("Function only supports up to the fourth moment")
    end

    out = zeros(4, 4)
    out[1, 1] = 1

    out[1, 2] = -2 * μ[1]
    out[2, 2] = 1.0

    out[1, 3] = (3 * μ[1] * μ[3] - 3 * μ[2]^2) / ((μ[2] - μ[1]^2)^(5 / 2))
    out[2, 3] = 3 * (μ[1] * μ[2] - μ[3]) / (2 * (μ[2] - μ[1]^2)^(5 / 2))
    out[3, 3] = 1 / ((μ[2] - μ[1]^2)^(3 / 2))

    out[1, 4] = -4 * (3 * μ[1]^2 * μ[3] - μ[1] * (3 * μ[2]^2 + μ[4]) + μ[2] * μ[3]) / ((μ[2] - μ[1]^2)^3)
    out[2, 4] = -2 * (3 * μ[1]^2 * μ[2] - 4 * μ[1] * μ[3] + μ[4]) / ((μ[2] - μ[1]^2)^3)
    out[3, 4] = -4 * μ[1] / ((μ[2] - μ[1]^2)^2)
    out[4, 4] = 1 / ((μ[2] - μ[1]^2)^2)

    return out[1:n, 1:n]
end

# Specific functions

function MTPder(μ::Vector{T}, d::Beta)::Matrix{Float64} where {T<:Real}
    out = zeros(2, 2)
    # ∂α/∂μ
    out[1, 1] = -μ[2] * (μ[1]^2 - 2 * μ[1] + μ[2]) / ((μ[2] - μ[1]^2)^2)
    out[2, 1] = -(1 - μ[1]) * μ[1]^2 / ((μ[2] - μ[1]^2)^2)
    # ∂β/∂μ
    out[1, 2] = -out[1, 1] + (μ[1]^2 - 2 * μ[1] * μ[2] + μ[2]) / ((μ[2] - μ[1]^2)^2)
    out[2, 2] = -out[2, 1] + (μ[1] - 1) * μ[1] / ((μ[2] - μ[1]^2)^2)

    return out
end

function MTPder(μ::Vector{T}, d::BetaBinomial)::Matrix{Float64} where {T<:Real}
    out = zeros(2, 2)
    n = d.n # fixed
    # ∂α/∂μ
    out[1, 1] = -(μ[1]^2 * ((n - 1) * μ[2] + n^2) - 2 * n^2 * μ[1] * μ[2] + n * μ[2]^2) / (((n - 1) * μ[1]^2 + n * μ[1] - n * μ[2])^2)
    out[2, 1] = ((n - 1) * μ[1]^2 * (μ[1] - n)) / (((n - 1) * μ[1]^2 + n * μ[1] - n * μ[2])^2)
    # ∂β/∂μ
    out[1, 2] = (n - 1) * n * (n * μ[1]^2 - 2 * μ[1] * μ[2] + n * μ[2]) / (((n - 1) * μ[1]^2 + n * μ[1] - n * μ[2])^2) - out[1, 1]
    out[2, 2] = (n - 1) * n * μ[1] * (μ[1] - n) / (((n - 1) * μ[1]^2 + n * μ[1] - n * μ[2])^2) - out[2, 1]

    return out
end

function MTPder(μ::Vector{T}, d::BetaPrime)::Matrix{Float64} where {T<:Real}
    out = zeros(2, 2)
    # ∂α/∂μ
    out[1, 1] = μ[2] * (μ[1]^2 + 2 * μ[1] + μ[2]) / ((μ[2] - μ[1]^2)^2)
    out[2, 1] = -μ[1]^2 * (μ[1] + 1) / ((μ[2] - μ[1]^2)^2)
    # ∂β/∂μ
    out[1, 2] = (μ[1] * out[1, 1] - μ[1] * (μ[1] + μ[2]) / (μ[2] - μ[1]^2)) / (μ[1]^2)
    out[2, 2] = out[2, 1] / μ[1]

    return out
end

function MTPder(μ::Vector{T}, d::Binomial)::Matrix{Float64} where {T<:Real}
    out = zeros(1, 1)
    n = d.n # fixed
    out[1, 1] = 1 / n
    return out
end

# Use fallback function
# function MTPder(μ::Vector{T}, d::Chi)::Matrix{Float64} where {T <: Real}
#     out = zeros(1, 1)
#     out[1, 1] = Calculus.derivative(x -> MTP([x], d), μ[1])[1]
#     return out
# end

function MTPder(μ::Vector{T}, d::Chisq)::Matrix{Float64} where {T<:Real}
    out = zeros(1, 1)
    out[1, 1] = 1.0
    return out
end

function MTPder(μ::Vector{T}, d::Erlang)::Matrix{Float64} where {T<:Real}
    out = zeros(1, 1)
    α = d.α # fixed
    out[1, 1] = 1 / α
    return out
end

function MTPder(μ::Vector{T}, d::Exponential)::Matrix{Float64} where {T<:Real}
    out = zeros(1, 1)
    out[1, 1] = 1.0
    return out
end

function MTPder(μ::Vector{T}, d::FDist)::Matrix{Float64} where {T<:Real}
    out = zeros(2, 2)

    # ∂ν2/∂μ
    out[1, 2] = -2 / ((μ[1] - 1)^2)
    out[2, 2] = 0.0

    # ∂ν1/∂μ
    ν2 = 2 * μ[1] / (μ[1] - 1)
    temp1 = ν2^2 * ((μ[1]^2 - μ[2]) * out[1, 2] - 12 * μ[1])
    temp2 = 2 * ν2 * ((μ[1]^2 - μ[2]) * out[1, 2] + 8 * μ[1])
    temp3 = 16 * (μ[2] - μ[1]^2) * out[1, 2] + 2 * μ[1] * ν2^3
    dc1 = -(ν2 - 2) * (temp1 + temp2 + temp3) / (2 * ν2^3)
    dc2 = (ν2 - 4) * (ν2 - 2)^2 / (2 * ν2^2)
    c = (μ[2] - μ[1]^2) * (ν2 - 2)^2 * (ν2 - 4) / (2 * ν2^2)

    out[1, 1] = ((c - 1) * out[1, 2] - (ν2 - 2) * dc1) / ((c - 1)^2)
    out[2, 1] = ((c - 1) * out[2, 2] - (ν2 - 2) * dc2) / ((c - 1)^2)

    return out
end

function MTPder(μ::Vector{T}, d::Gamma)::Matrix{Float64} where {T<:Real}
    out = zeros(2, 2)
    # ∂θ/∂μ
    out[1, 2] = -μ[2] / (μ[1]^2) - 1
    out[2, 2] = 1 / μ[1]
    # ∂α/∂μ
    θ = (μ[2] - μ[1]^2) / μ[1]
    out[1, 1] = (θ - μ[1] * out[1, 2]) / (θ^2)
    out[2, 1] = -μ[1] * out[2, 2] / (θ^2)

    return out
end

# Use fallback function
# function MTPder(μ::Vector{T}, d::GeneralizedExtremeValue)::Matrix{Float64} where {T <: Real}
#
# end

# Use fallback function
# function MTPder(μ::Vector{T}, d::GeneralizedPareto)::Matrix{Float64} where {T <: Real}
#
# end

function MTPder(μ::Vector{T}, d::Geometric)::Matrix{Float64} where {T<:Real}
    out = zeros(1, 1)
    out[1, 1] = -1 / ((μ[1] + 1)^2)

    return out
end

function MTPder(μ::Vector{T}, d::Gumbel)::Matrix{Float64} where {T<:Real}
    out = zeros(2, 2)
    γ = Base.MathConstants.eulergamma
    # ∂θ/∂μ
    out[1, 2] = -sqrt(6) * μ[1] / (π^2 * sqrt((μ[2] - μ[1]^2) / (π^2)))
    out[2, 2] = sqrt(3 / 2) / (π^2 * sqrt((μ[2] - μ[1]^2) / (π^2)))
    # ∂μd/∂μ
    out[1, 1] = 1.0 - γ * out[1, 2]
    out[2, 1] = -γ * out[2, 2]

    return out
end

function MTPder(μ::Vector{T}, d::InverseGamma)::Matrix{Float64} where {T<:Real}
    out = zeros(2, 2)
    # ∂θ/∂μ
    out[1, 2] = μ[2] * (μ[2] + μ[1]^2) / ((μ[2] - μ[1]^2)^2)
    out[2, 2] = -μ[1]^3 / ((μ[2] - μ[1]^2)^2)

    # ∂α/∂μ
    θ = μ[1] * μ[2] / (μ[2] - μ[1]^2)
    out[1, 1] = (μ[1] * out[1, 2] - θ) / (μ[1]^2)
    out[2, 1] = out[2, 2] / μ[1]

    return out
end

function MTPder(μ::Vector{T}, d::InverseGaussian)::Matrix{Float64} where {T<:Real}
    out = zeros(2, 2)

    # ∂μd/∂μ
    out[1, 1] = 1.0
    out[2, 1] = 0.0

    # ∂λ/∂μ
    out[1, 2] = (3 * μ[1]^2 * μ[2] - μ[1]^4) / ((μ[2] - μ[1]^2)^2)
    out[2, 2] = -μ[1]^3 / ((μ[2] - μ[1]^2)^2)

    return out
end

function MTPder(μ::Vector{T}, d::Laplace)::Matrix{Float64} where {T<:Real}
    out = zeros(2, 2)

    # ∂μd/∂μ
    out[1, 1] = 1.0
    out[2, 1] = 0.0

    # ∂θ/∂μ
    out[1, 2] = -μ[1] / sqrt(2 * (μ[2] - μ[1]^2))
    out[2, 2] = 1 / (2 * sqrt(2 * (μ[2] - μ[1]^2)))

    return out
end

function MTPder(μ::Vector{T}, d::LogNormal)::Matrix{Float64} where {T<:Real}
    out = zeros(2, 2)

    # ∂σ/∂μ
    out[1, 2] = -1 / (μ[1] * sqrt(log(μ[2]) - 2 * log(μ[1])))
    out[2, 2] = 1 / (2 * μ[2] * sqrt(log(μ[2]) - 2 * log(μ[1])))

    # ∂μd/∂μ
    σ = sqrt(log(μ[2]) - 2 * log(μ[1]))
    out[1, 1] = -2 * σ * out[1, 2]
    out[2, 1] = 1 / (2 * μ[2]) - 2 * σ * out[2, 2]

    return out
end

function MTPder(μ::Vector{T}, d::Logistic)::Matrix{Float64} where {T<:Real}
    out = zeros(2, 2)

    # ∂μd/∂μ
    out[1, 1] = 1.0
    out[2, 1] = 0.0

    # ∂θ/∂μ
    out[1, 2] = -sqrt(3) * μ[1] / (π^2 * sqrt((μ[2] - μ[1]^2) / (π^2)))
    out[2, 2] = sqrt(3) / (2 * π^2 * sqrt((μ[2] - μ[1]^2) / (π^2)))

    return out
end

function MTPder(μ::Vector{T}, d::NegativeBinomial)::Matrix{Float64} where {T<:Real}
    out = zeros(2, 2)

    # ∂p/∂μ
    out[1, 2] = (μ[1]^2 + μ[2]) / ((μ[2] - μ[1]^2)^2)
    out[2, 2] = -μ[1] / ((μ[2] - μ[1]^2)^2)

    # ∂r/∂μ
    p = μ[1] / (μ[2] - μ[1]^2)
    out[1, 1] = (μ[1] * out[1, 2] - p^2 + p) / ((1 - p)^2)
    out[2, 1] = μ[1] * out[2, 2] / ((1 - p)^2)

    return out
end

function MTPder(μ::Vector{T}, d::NoncentralChisq)::Matrix{Float64} where {T<:Real}
    out = zeros(2, 2)

    # ∂ν/∂μ
    out[1, 1] = 2 + μ[1]
    out[2, 1] = -1 / 2

    # ∂λ/∂μ
    out[1, 2] = -μ[1] - 1
    out[2, 2] = 1 / 2

    return out
end

# Use fallback function
# function MTPder(μ::Vector{T}, d::NoncentralF)::Matrix{Float64} where {T <: Real}
#
# end

# Use fallback function
# function MTPder(μ::Vector{T}, d::NoncentralT)::Matrix{Float64} where {T <: Real}
#
# end

function MTPder(μ::Vector{T}, d::Normal)::Matrix{Float64} where {T<:Real}
    out = zeros(2, 2)

    # ∂μd/∂μ
    out[1, 1] = 1.0
    out[2, 1] = 0.0

    # ∂σ/∂μ
    out[1, 2] = -μ[1] / sqrt(μ[2] - μ[1]^2)
    out[2, 2] = 1 / (2 * sqrt(μ[2] - μ[1]^2))

    return out
end

function MTPder(μ::Vector{T}, d::NormalCanon)::Matrix{Float64} where {T<:Real}
    out = zeros(2, 2)

    # ∂λ/∂μ
    out[1, 2] = 2 * μ[1] / ((μ[2] - μ[1]^2)^2)
    out[2, 2] = -1 / ((μ[2] - μ[1]^2)^2)

    # ∂η/∂μ
    λ = 1 / (μ[2] - μ[1]^2)
    out[1, 1] = μ[1] * out[1, 2] + λ
    out[2, 1] = μ[1] * out[2, 2]

    return out
end


# Use fallback function
# function MTPder(μ::Vector{T}, d::NormalInverseGaussian)::Matrix{Float64} where {T <: Real}
#
# end

# Use fallback function
# function MTPder(μ::Vector{T}, d::PGeneralizedGaussian)::Matrix{Float64} where {T <: Real}
#
# end

function MTPder(μ::Vector{T}, d::Pareto)::Matrix{Float64} where {T<:Real}
    out = zeros(1, 1)
    θ = d.θ # Fixed

    out[1, 1] = -θ / ((μ[1] - θ)^2)

    return out
end

function MTPder(μ::Vector{T}, d::Poisson)::Matrix{Float64} where {T<:Real}
    out = zeros(1, 1)
    out[1, 1] = 1.0

    return out
end

function MTPder(μ::Vector{T}, d::Rayleigh)::Matrix{Float64} where {T<:Real}
    out = zeros(1, 1)
    out[1, 1] = 1 / sqrt(π / 2)

    return out
end

function MTPder(μ::Vector{T}, d::Skellam)::Matrix{Float64} where {T<:Real}
    out = zeros(2, 2)

    # ∂μ1/∂μ
    out[1, 1] = 1 / 2 - μ[1]
    out[2, 1] = 1 / 2

    # ∂μ1/∂μ
    out[1, 2] = out[1, 1] - 1
    out[2, 2] = out[2, 1]

    return out
end

function MTPder(μ::Vector{T}, d::TDist)::Matrix{Float64} where {T<:Real}
    out = zeros(1, 1)
    out[1, 1] = -2 / ((μ[1] - 1)^2)

    return out
end

# Use fallback function
# function MTPder(μ::Vector{T}, d::VonMises)::Matrix{Float64} where {T <: Real}
#
# end

# Use fallback function
# function MTPder(μ::Vector{T}, d::Weibull)::Matrix{Float64} where {T <: Real}
#
# end
