# Fallback that always returns true
function checkParam(d::T, θ::T2)::Bool where {T <: UnivariateDistribution, T2 <: Real}
    return true
end

function checkParam(d::T, θ::Vector{T2})::Bool where {T <: UnivariateDistribution, T2 <: Real}
    return true
end

function checkParam(d::Beta, θ::Vector{T})::Bool where {T <: Real}
    ((θ[1] > 0) & (θ[2] > 0)) ? true : false
end

# Input θ: only α and β, since n fixed
function checkParam(d::BetaBinomial, θ::Vector{T})::Bool where {T <: Real}
    ((θ[1] > 0) & (θ[2] > 0)) ? true : false
end

function checkParam(d::BetaPrime, θ::Vector{T})::Bool where {T <: Real}
    ((θ[1] > 0) & (θ[2] > 0)) ? true : false
end


function checkParam(d::Binomial, θ::T)::Bool where {T <: Real}
    0 <= θ <= 1 ? true : false
end

function checkParam(d::Chi, θ::T)::Bool where {T <: Real}
    θ > 0 ? true : false
end

function checkParam(d::Chisq, θ::T)::Bool where {T <: Real}
    θ > 0 ? true : false
end

function checkParam(d::Erlang, θ::T)::Bool where {T <: Real}
    # α fixed
    θ > 0 ? true : false
end

function checkParam(d::Exponential, θ::T)::Bool where {T <: Real}
    θ > 0 ? true : false
end

function checkParam(d::FDist, θ::Vector{T})::Bool where {T <: Real}
    ((θ[1] > 0) & (θ[2] > 0)) ? true : false
end

function checkParam(d::Gamma, θ::Vector{T})::Bool where {T <: Real}
    ((θ[1] > 0) & (θ[2] > 0)) ? true : false
end

function checkParam(d::GeneralizedExtremeValue, θ::Vector{T})::Bool where {T <: Real}
    θ[2] > 0 ? true : false
end

function checkParam(d::GeneralizedPareto, θ::Vector{T})::Bool where {T <: Real}
    θ[2] > 0 ? true : false
end

function checkParam(d::Geometric, θ::T)::Bool where {T <: Real}
    0 <= θ <= 1 ? true : false
end

function checkParam(d::Gumbel, θ::Vector{T})::Bool where {T <: Real}
    θ[2] > 0 ? true : false
end

function checkParam(d::InverseGamma, θ::Vector{T})::Bool where {T <: Real}
    ((θ[1] > 0) & (θ[2] > 0)) ? true : false
end

function checkParam(d::InverseGaussian, θ::Vector{T})::Bool where {T <: Real}
    θ[2] > 0 ? true : false
end


function checkParam(d::Laplace, θ::Vector{T})::Bool where {T <: Real}
    θ[2] > 0 ? true : false
end

function checkParam(d::LogNormal, θ::Vector{T})::Bool where {T <: Real}
    θ[2] > 0 ? true : false
end

function checkParam(d::Logistic, θ::Vector{T})::Bool where {T <: Real}
    θ[2] > 0 ? true : false
end


function checkParam(d::NegativeBinomial, θ::Vector{T})::Bool where {T <: Real}
    (θ[1] > 0) & (0 <= θ[2] <= 1) ? true : false
end

function checkParam(d::NoncentralChisq, θ::Vector{T})::Bool where {T <: Real}
    ((θ[1] > 0) & (θ[2] > 0)) ? true : false
end

function checkParam(d::NoncentralF, θ::Vector{T})::Bool where {T <: Real}
    all(θ .> 0) ? true : false
end

function checkParam(d::NoncentralT, θ::Vector{T})::Bool where {T <: Real}
    ((θ[1] > 0) & (θ[2] > 0)) ? true : false
end

function checkParam(d::Normal, θ::Vector{T})::Bool where {T <: Real}
    θ[2] > 0 ? true : false
end

function checkParam(d::NormalCanon, θ::Vector{T})::Bool where {T <: Real}
    θ[2] > 0 ? true : false
end

function checkParam(d::NormalInverseGaussian, θ::Vector{T})::Bool where {T <: Real}
    (θ[2]^2 > θ[3]^2) ? true : false
end

# Symmetric Distribution, therefore also include kurtosis
function checkParam(d::PGeneralizedGaussian, θ::Vector{T})::Bool where {T <: Real}
    ((θ[2] > 0) & (θ[3] > 0)) ? true : false
end

function checkParam(d::Pareto, θ::T)::Bool where {T <: Real}
    θ > 0 ? true : false
end

function checkParam(d::Poisson, θ::T)::Bool where {T <: Real}
    θ > 0 ? true : false
end

function checkParam(d::Rayleigh, θ::T)::Bool where {T <: Real}
    θ > 0 ? true : false
end

function checkParam(d::Skellam, θ::Vector{T})::Bool where {T <: Real}
    (θ[1] > 0) & (θ[2] > 0) ? true : false
end

function checkParam(d::TDist, θ::T)::Bool where {T <: Real}
    θ > 0 ? true : false
end

function checkParam(d::GeneralizedTDist, θ::T)::Bool where {T<:Real}
    (θ[2] > 0) & (θ[3] > 0) ? true : false
end

function checkParam(d::VonMises, θ::Vector{T})::Bool where {T <: Real}
    θ[2] > 0 ? true : false
end

function checkParam(d::Weibull, θ::Vector{T})::Bool where {T <: Real}
    (θ[1] > 0) & (θ[2] > 0) ? true : false
end
