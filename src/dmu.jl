# For some distributions, computes derivative of mean with respect to parameters
# Fallback gives derivatives of the first moments, like PTMder similar to MTPder

function dμ(d::Beta{Float64})::Matrix{Float64}
    α = d.α
    β = d.β
    μ = α / (α + β)
    dμ = [β / (α + β)^2; -α / (α + β)^2]
    dσ = [β * (-2 * α^2 - α * (β + 1) + β * (β + 1)) / ((α + β)^2 * (α + β + 1)^2)
        α * (α^2 - α * β + α - β * (2 * β + 1)) / ((α + β)^3 * (α + β + 1)^2)]
    [dμ dσ + 2 * μ .* dμ]
end

function dμ(d::BetaBinomial{Float64})::Matrix{Float64}
    n = d.n # Fixed
    α = d.α
    β = d.β
    μ = n * α / (α + β)
    dμ = [n * β / (α + β)^2; -n * α / (α + β)^2]
    dσ = [n * β * (-α^2 - α^2 * (β + 2 * n) + α * (β + 1) * (β - n) + β * (β + 1) * (β + n))
        n * α * (α^3 + α^2 * (β + n + 1) - α * (β - 1) * (β + n) - β * (β^2 + 2 * β * n + n))] ./ ((α + β)^3 * (α + β + 1)^2)
    [dμ dσ + 2 * μ .* dμ]
end

function dμ(d::BetaPrime{Float64})::Matrix{Float64}
    α = d.α
    β = d.β
    μ = α / (β - 1)
    dμ = [1 / (β - 1); -α / ((β - 1)^2)]
    dσ = [(2 * α + β - 1) / ((β - 2) * (β - 1)^2)
        -α * (α * (3 * β - 5) + 2 * β^2 - 5 * β + 3) / ((β - 2)^2 * (β - 1)^3)]
    [dμ dσ + 2 * μ .* dμ]
end


function dμ(d::Binomial{Float64})::Matrix{Float64}
    # Fixed n
    reshape([d.n + 0.0], (1, 1))
end

function dμ(d::Chi{Float64})::Matrix{Float64}
    ν = d.ν
    i1 = (ν + 1) / 2
    i2 = ν / 2
    reshape([gamma(i1) * (digamma(i1) - digamma(i2)) / (sqrt(2) * gamma(i2))], (1, 1))
end

function dμ(d::Chisq{Float64})::Matrix{Float64}
    reshape([1.0], (1, 1))
end

function dμ(d::Erlang{Float64})::Matrix{Float64}
    # α fixed
    reshape([d.θ], (1, 1))
end

function dμ(d::Exponential{Float64})::Matrix{Float64}
    reshape([1.0], (1, 1))
end

function dμ(d::FDist{Float64})::Matrix{Float64}
    ν1 = d.ν1
    ν2 = d.ν2


    dμ = [0; -2 / (ν2 - 2)^2]
    μ = ν2 / (ν2 - 2)

    dσ = [-2 * ν2^2 / (ν1^2 * (ν2 - 2) * (ν2 - 4))
        -2 * ν2 * (ν1 * (ν2^2 + 2 * ν2 - 16) + 6 * ν2^2 - 28 * ν2 + 32) / (ν1 * (ν2 - 2)^3 * (ν2 - 4)^2)]
    [dμ dσ + 2 * μ .* dμ]
end

function dμ(d::Gamma{Float64})::Matrix{Float64}
    α = d.α
    θ = d.θ

    dμ = [θ; α]
    dσ = [θ^2; 2 * α * θ]
    μ = α * θ
    [dμ dσ + 2 * μ .* dμ]
end

function dμ(d::GeneralizedExtremeValue{Float64})::Matrix{Float64}
    μ = d.μ
    σ = d.σ
    ξ = d.ξ
    g1 = gamma(1 - ξ)
    g2 = gamma(1 - 2 * ξ)
    g3 = gamma(1 - 3 * ξ)

    dμ = [1; (g1 - 1) / ξ; (σ - g1 * (σ * ξ * digamma(1 - ξ) + σ)) / ξ^2]
    dσ = [0; 2 * σ * (g2 - g1^2) / ξ^2
        -2 * σ^2 * (g2 * (ξ * digamma(1 - 2 * ξ) + 1) - g1^2 * (ξ * digamma(1 - ξ) + 1)) / ξ^3]
    μd = mean(d)
    σd = var(d)
    γ = skewness(d)

    dg1 = -g1 * digamma(1 - ξ)
    dg2 = -2 * g2 * digamma(1 - 2 * ξ)
    dg3 = -3 * g3 * digamma(1 - 3 * ξ)
    dγ = [0; 0; (-6 * g2^2 * dg1 + 6 * g1 * g3 * dg1 + g2 * (3 * g1 * dg2 + 2 * dg3) - 2 * g1^2 * dg3 - 3 * g3 * dg2) / (2 * (g2 - g1^2)^(5 / 2))]

    out2 = dσ + 2 * μd .* dμ
    out3 = σd^(3 / 2) .* dγ .+ (3 / 2 * sqrt(σd) * γ) .* dσ + 3 * μd .* (dσ .- μd .* dμ) + (3 * σd) .* dμ

    [dμ out2 out3]
end

function dμ(d::GeneralizedPareto{Float64})::Matrix{Float64}
    μ = d.μ
    σ = d.σ
    ξ = d.ξ

    dμ = [1; 1 / (1 - ξ); σ / (1 - ξ)^2]
    dσ = [0; 2 * σ / ((1 - ξ)^2 * (1 - 2 * ξ)); -2 * σ^2 * (3 * ξ - 2) / ((1 - 2 * ξ)^2 * (1 - ξ)^3)]

    μd = mean(d)
    σd = var(d)
    γ = skewness(d)

    dγ = [0; 0; 6 * (ξ - 1)^2 / ((1 - 3 * ξ)^2 * sqrt(1 - 2 * ξ))]

    out2 = dσ + 2 * μd .* dμ
    out3 = σd^(3 / 2) .* dγ .+ (3 / 2 * sqrt(σd) * γ) .* dσ + 3 * μd .* (dσ .- μd .* dμ) + (3 * σd) .* dμ

    [dμ out2 out3]
end

function dμ(d::Geometric{Float64})::Matrix{Float64}
    reshape([-1 / d.p^2], (1, 1))
end

function dμ(d::Gumbel{Float64})::Matrix{Float64}
    μ = d.μ
    θ = d.θ
    γ = Base.MathConstants.eulergamma

    dμ = [1; γ]
    dσ = [0; 2 * π^2 * θ / 6]
    μd = mean(d)
    σ = var(d)

    [dμ dσ + 2 * μd .* dμ]
end

function dμ(d::InverseGamma{Float64})::Matrix{Float64}
    α = d.invd.α
    θ = d.θ

    dμ = [-θ / ((α - 1)^2); 1 / (α - 1)]
    dσ = [(5 - 3 * α) * θ^2 / ((α - 2)^2 * (α - 1)^3); 2 * θ / ((α - 1)^2 * (α - 2))]

    μ = mean(d)
    σ = var(d)

    [dμ dσ + 2 * μ .* dμ]
end

function dμ(d::InverseGaussian{Float64})::Matrix{Float64}
    μ = d.μ
    λ = d.λ

    dμ = [1; 0]
    dσ = [3 * μ^2 / λ; -μ^3 / (λ^2)]
    μd = mean(d)

    [dμ dσ + 2 * μd .* dμ]
end


function dμ(d::Laplace{Float64})::Matrix{Float64}
    μ = d.μ
    θ = d.θ

    dμ = [1; 0]
    dσ = [0; 4 * θ]
    μd = mean(d)

    [dμ dσ + 2 * μd .* dμ]
end

function dμ(d::LogNormal{Float64})::Matrix{Float64}
    μ = d.μ
    σ = d.σ

    μd = mean(d)
    σd = var(d)
    dμ = [μd; μd * σ]
    dσ = [2 * σd; 2 * (2 * exp(σ^2) - 1) * σ * exp(2 * μ + σ^2)]

    [dμ dσ + 2 * μd .* dμ]
end

function dμ(d::Logistic{Float64})::Matrix{Float64}
    μ = d.μ
    θ = d.θ

    dμ = [1; 0]
    dσ = [0; 2 * θ * π^2 / 3]
    μd = mean(d)
    [dμ dσ + 2 * μd .* dμ]
end


function dμ(d::NegativeBinomial{Float64})::Matrix{Float64}
    r = d.r
    p = d.p

    dμ = [(1 - p) / p; -r / p^2]
    dσ = [(1 - p) / p^2; r * (p - 2) / p^3]
    μd = mean(d)
    [dμ dσ + 2 * μd .* dμ]
end

function dμ(d::NoncentralChisq{Float64})::Matrix{Float64}
    ν = d.ν
    λ = d.λ

    dμ = [1.0; 1]
    dσ = [2; 4]
    μd = mean(d)
    [dμ dσ + 2 * μd .* dμ]
end

function dμ(d::NoncentralF{Float64})::Matrix{Float64}
    ν1 = d.ν1
    ν2 = d.ν2
    λ = d.λ

    dμ = [-λ * ν2 / (ν1^2 * (ν2 - 2))
        -2 * (ν1 + λ) / (ν1 * (ν2 - 2)^2)
        ν2 / (ν1 * (ν2 - 2))]
    temp1 = ν1 * (ν2 + 2 * λ - 2) + 2 * λ * (2 * ν2 + λ - 4)
    temp2 = ν1^2 * (ν2^2 + 2 * ν2 - 16) + 2 * ν1 * (ν2^2 * (λ + 3) + 2 * ν2 * (λ - 7) - 16 * (λ - 1))
    temp2 += λ * (ν2^2 * (λ + 12) + 2 * ν2 * (λ - 28) - 16 * (λ - 4))

    dσ = [-2 * ν2^2 * temp1 / ((ν2 - 2)^2 * ν1^3 * (ν2 - 4))
        -2 * ν2 * temp2 / ((ν2 - 2)^3 * ν1^2 * (ν2 - 4)^2)
        4 * ν2^2 * (ν1 + ν2 + λ - 2) / ((ν2 - 2)^2 * ν1^2 * (ν2 - 4))]

    μd = mean(d)

    out3 = zeros(3)
    out3[1] = -ν2^3 * (3 * λ^3 + 6 * λ^2 * (ν1 + 6) + 3 * λ * (ν1^2 + 12 * ν1 + 24) + 2 * ν1 * (3 * ν1 + 8))
    out3[1] = out3[1] / (ν1^4 * (ν2 - 6) * (ν2 - 4) * (ν2 - 2))
    out3[2] = -((4 * ν2^2 * (3 * ν2^2 - 22 * ν2 + 36) * (λ^3 + 3 * λ^2 * (ν1 + 4) + (3 * λ + ν1) * (ν1^2 + 6 * ν1 + 8))))
    out3[2] = out3[2] / (ν1^3 * (ν2 - 6)^2 * (ν2 - 4)^2 * (ν2 - 2)^2)
    out3[3] = 3 * ν2^3 * (λ^2 + 2 * λ * (ν1 + 4) + ν1^2 + 6 * ν1 + 8)
    out3[3] = out3[3] / (ν1^3 * (ν2 - 6) * (ν2 - 4) * (ν2 - 2))

    [dμ dσ + 2 * μd .* dμ out3]
end

function dμ(d::NoncentralT{Float64})::Matrix{Float64}
    ν = d.ν
    λ = d.λ

    g1 = gamma((ν - 1) / 2)
    g2 = gamma(ν / 2)
    dg1 = digamma((ν - 1) / 2)
    dg2 = digamma(ν / 2)

    dμ = [λ * g1 * (ν * (dg1) - ν * dg2 + 1) / (2 * sqrt(2) * sqrt(ν) * g2); sqrt(ν / 2) * g1 / g2]
    dσ = [-(λ^2 + 1) * ν / (ν - 2)^2 + (λ^2 + 1) / (ν - 2) - λ^2 * g1^2 / (2 * g2^2) - λ^2 * ν * g1^2 * dg1 / (2 * g2^2) + λ^2 * ν * g1^2 * dg2 / (2 * g2^2)
        2 * λ * ν / (ν - 2) - 2 * λ * ν / 2 * (g1 / g2)^2]

    μd = mean(d)

    [dμ dσ + 2 * μd .* dμ]
end

function dμ(d::Normal{Float64})::Matrix{Float64}
    μ = d.μ
    σ = d.σ

    dμ = [1; 0]
    dσ = [0; 2 * σ]

    μd = mean(d)

    [dμ dσ + 2 * μd .* dμ]
end

function dμ(d::NormalCanon{Float64})::Matrix{Float64}
    η = d.η
    λ = d.λ

    dμ = [1 / λ; -η / λ^2]
    dσ = [0; -1 / λ^2]
    μd = mean(d)

    [dμ dσ + 2 * μd .* dμ]
end

function dμ(d::NormalInverseGaussian{Float64})::Matrix{Float64}
    μ = d.μ
    α = d.α
    β = d.β
    δ = d.δ

    g = sqrt(α^2 - β^2)

    dμ = [1; -α * β * δ / (g^3); α^2 * δ / (g^3); β / g]
    dσ = [0; -δ * (α^3 - 2 * α * β^2) / (g^5); 3 * α^2 * β * δ / (g^5); α^2 / (g^3)]

    dγ = [0 (6 * β^3 - 9 * α^2 * β) / (2 * α^2 * g^2 * sqrt(δ * g))
        3 * (2 * α^2 - β^2) / (2 * (α^3 - α * β^2) * sqrt(δ * g))
        -3 * β / (2 * α * δ * sqrt(δ * g))] # derivative of skewness
    γ = skewness(d)
    dκ = [0 -3 * (α^4 + 9 * α^2 * β^2 - 6 * β^4) / (α^3 * δ * g^3)
        3 * β * (7 * α^2 - 3 * β^2) / (α^2 * δ * g^3)
        3 * (3 * β^2 / α^2 + 1) / (δ^2 * g)]
    κ = kurtosis(d, false)

    μd = mean(d)
    σd = var(d)
    Ex2 = σd + μd^2
    Ex3 = σd^(3 / 2) * γ + 3 * μd * σd - μd^3

    out2 = dσ + 2 * μd .* dμ
    out3 = σd^(3 / 2) .* dγ .+ (3 / 2 * sqrt(σd) * γ) .* dσ + 3 * μd .* (dσ .- μd .* dμ) + (3 * σd) .* dμ
    out4 = dκ .* σd^2 .+ (2 * κ * σd) .* dσ .+ (4 * Ex3) .* dμ .+ (4 * μd) .* out3 .- (12 * μd * Ex2) .* dμ - (6 * μd^2) .* out2 .- (12 * μd^3) .* dμ

    [dμ out2 out3 out4]
end

# Symmetric Distribution, therefore also include kurtosis
function dμ(d::PGeneralizedGaussian{T1,T2,T3})::Matrix{Float64} where {T1,T2,T3<:Real}
    μ = d.μ
    α = d.α
    p = d.p

    μd = mean(d)
    σ = std(d)

    dμ = [1; 0; 0]
    dσ = [0; 2 * σ^2 / α; σ^2 * (digamma(1 / p) - 3 * digamma(3 / p)) / p^2]

    μd = mean(d)
    σd = var(d)
    γ = skewness(d)
    κ = kurtosis(d, false)

    dγ = [0; 0; 0]
    dκ = [0; 0; -gamma(1 / p) * gamma(5 / p) * (digamma(1 / p) - 6 * digamma(3 / p) + 5 * digamma(5 / p)) / (p^2 * gamma(3 / p)^2)]
    out2 = dσ + 2 * μd .* dμ
    out3 = σd^(3 / 2) .* dγ .+ (3 / 2 * sqrt(σd) * γ) .* dσ + 3 * μd .* (dσ .- μd .* dμ) + (3 * σd) .* dμ

    Ex2 = σd + μd^2
    Ex3 = σd^(3 / 2) * γ + 3 * μd * σd - μd^3
    Ex4 = κ * σd^2 + 4 * μd * Ex3 - 6 * μd^2 * Ex2 + 3 * μd^4
    out4 = dκ .* σd^2 .+ (2 * κ * σd) .* dσ .+ (4 * Ex3) .* dμ .+ (4 * μd) .* out3 .- (12 * μd * Ex2) .* dμ - (6 * μd^2) .* out2 .- (12 * μd^3) .* dμ

    [dμ out2 out3 out4]
end

function dμ(d::Pareto{Float64})::Matrix{Float64}
    α = d.α
    θ = d.θ # Fixed

    reshape([-θ / (α - 1)], (1, 1))
end

function dμ(d::Poisson{Float64})::Matrix{Float64}
    reshape([d.λ], (1, 1))
end

function dμ(d::Rayleigh{Float64})::Matrix{Float64}
    reshape([sqrt(π / 2)], (1, 1))
end

function dμ(d::Skellam{Float64})::Matrix{Float64}
    μ1 = d.μ1
    μ2 = d.μ2

    dμ = [1.0, -1]
    dσ = [1, 1]
    μd = μ1 - μ2

    return [dμ dσ + 2 * μd .* dμ]
end

# Symmetric around zero, therefore only use E(X^2)
function dμ(d::TDist{Float64})::Matrix{Float64}
    ν = d.ν
    return reshape([-2 / (ν - 2)^2], (1, 1))
end

function dμ(d::VonMises{Float64})::Matrix{Float64}
    μ = d.μ
    κ = d.κ

    dμ = [1; 0]
    dσ = [0; -(besseli(0, κ)^2 + besseli(2, κ) * besseli(0, κ) - 2 * besseli(1, κ)^2) / (2 * besseli(0, κ)^2)]
    μd = mean(d)

    return [dμ dσ + 2 * μd .* dμ]
end

function dμ(d::Weibull{Float64})::Matrix{Float64}
    α = d.α
    θ = d.θ

    g1 = gamma(1 + 1 / α)
    g2 = gamma(1 + 2 / α)
    dg1 = digamma(1 + 1 / α)
    dg2 = digamma(1 + 2 / α)
    μd = mean(d)

    dμ = [-μd * dg1 / α^2; g1]
    dσ = [2 * θ^2 * (g1^2 * dg1 - g2 * dg2) / α^2; 2 * θ * (g2 - g1^2)]

    return [dμ dσ + 2 * μd .* dμ]
end

#########################
### Fallback function ###
#########################

function dμ(d::T)::Matrix{Float64} where {T<:UnivariateDistribution}
    # Parameter names
    fn = propertynames(d)
    # Number of parameters
    nPar = nParEff(d)
    out = zeros(nPar, nPar)

    # Parameter vector
    θ = getParams(d)

    function tf(θ, d, δ, i, p)
        θtf = copy(θ)
        θtf[i] = δ
        dtf = NewDist(d, θtf)
        return mean(dPower(dtf, p))
    end

    for i = 1:nPar, j = 1:nPar
        out[i, j] = Calculus.derivative(δ -> tf(θ, d, δ, i, j), θ[i])
    end

    return out
end
