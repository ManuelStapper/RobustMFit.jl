### General functions for continuous and discrete distribution
function MTP(μ::Vector{T1}, d::T2)::Vector{Float64} where {T1 <: Real, T2 <: UnivariateDistribution}
    nPar = length(μ)
    function tf(θ, μ)
        dtf = try
            NewDist(d, θ)
        catch y
            if isa(y, ArgumentError) | isa(y, DomainError)
                return Inf
            end
        end
        μTheo = (i -> mean(dPower(dtf, i))).(1:nPar)
        if any(isnan.(μTheo))
            return Inf
        end
        sum((μTheo .- μ).^2)
    end
    par = (i -> params(d)[i]).(1:nPar)
    return optimize(vars -> tf(vars, μ), par .+ 0.0).minimizer
end

# Distribution-free function that returns Mean, Variance, Skewness, Kurtosis (not corrected)
# given raw moments
function MTP(μ::Vector{T1}) where {T1 <: Real}
    out = copy(μ)
    n = length(μ)

    if n > 4
        error("Function only supports up to the fourth moment")
    end

    if n >= 2
        out[2] = μ[2] - μ[1]^2
    end
    if n >= 3
        out[3] = (μ[3] - 3*μ[2]*μ[1] + 2*μ[1]^3)/(out[2]^(3/2))
    end
    if n == 4
        out[4] = (μ[4] - 4*μ[3]*μ[1] + 6*μ[2]*μ[1]^2 - 3*μ[1]^4)/(out[2]^2)
    end

    return out
end

# Specific functions
function MTP(μ::Vector{T}, d::Beta)::Vector{Float64} where {T <: Real}
    α = μ[1]*(μ[2] - μ[1])/(μ[1]^2 - μ[2])
    β = (μ[1] - 1)*(μ[1] - μ[2])/(μ[1]^2 - μ[2])
    return [α, β]
end

function MTP(μ::Vector{T}, d::BetaBinomial)::Vector{Float64} where {T <: Real}
    n = d.n # Fixed
    α = μ[1]*(μ[2] - μ[1]*n)/(μ[1]^2*(n-1) + μ[1]*n - μ[2]*n)
    β = (n - μ[1])*(μ[2] - μ[1]*n)/(μ[1]^2*(n-1) + μ[1]*n - μ[2]*n)
    return [α, β]
end

function MTP(μ::Vector{T}, d::BetaPrime)::Vector{Float64} where {T <: Real}
    α = μ[1]*(μ[1] + μ[2])/(μ[2] - μ[1]^2)
    β = α/μ[1] + 1
    return [α, β]
end

function MTP(μ::Vector{T}, d::Binomial)::Vector{Float64} where {T <: Real}
    n = d.n
    return [μ[1]/n]
end

function MTP(μ::Vector{T}, d::Chi)::Vector{Float64} where {T <: Real}
    # No analytical solution
    function tf(ν, c)
        if ν <= 0
            return Inf
        end
        c + logabsgamma(ν/2)[1] - logabsgamma((ν + 1)/2)[1]
    end
    return [find_zero(vars -> tf(vars, log(μ[1]) - log(2)/2), (0.01, exp(d.ν)), A42(), atol = 1e-05)]
end
# Gradient seems to be roughly 2*ν for large ν?

function MTP(μ::Vector{T}, d::Chisq)::Vector{Float64} where {T <: Real}
    return μ
end

function MTP(μ::Vector{T}, d::Erlang)::Vector{Float64} where {T <: Real}
    return [μ[1]/d.α]
end

function MTP(μ::Vector{T}, d::Exponential)::Vector{Float64} where {T <: Real}
    return μ
end

function MTP(μ::Vector{T}, d::FDist)::Vector{Float64} where {T <: Real}
    ν2 = 2*μ[1]/(μ[1] - 1)
    c = (μ[2] - μ[1]^2)*(ν2 - 2)^2*(ν2 - 4)/(2*ν2^2)
    ν1 = (ν2 - 2)/(c-1)
    return [ν1, ν2]
end

function MTP(μ::Vector{T}, d::Gamma)::Vector{Float64} where {T <: Real}
    θ = (μ[2] - μ[1]^2)/μ[1]
    α = μ[1]/θ
    return [α, θ]
end

function MTP(μ::Vector{T}, d::GeneralizedExtremeValue)::Vector{Float64} where {T <: Real}
    # No (easy) analytical solution
    function tf(θ, μ)
        if θ[3] > 1/3
            return Inf
        end
        dtf = try
            dtf = GeneralizedExtremeValue(θ[1], θ[2], θ[3])
        catch y
            if isa(y, ArgumentError) | isa(y, DomainError)
                return Inf
            end
        end
        μTheo = (i -> mean(dPower(dtf, i))).(1:3)
        if any(isnan.(μTheo))
            return Inf
        end
        sum((μTheo .- μ[1:3]).^2)
    end
    par = (i -> params(d)[i]).(1:3)
    return optimize(vars -> tf(vars, μ), par).minimizer
end

function MTP(μ::Vector{T}, d::GeneralizedPareto)::Vector{Float64} where {T <: Real}
    # No (easy) analytical solution
    function tf(θ, μ)
        if θ[3] > 1/3
            return Inf
        end
        dtf = try
            dtf = GeneralizedPareto(θ[1], θ[2], θ[3])
        catch y
            if isa(y, ArgumentError) | isa(y, DomainError)
                return Inf
            end
        end
        μTheo = (i -> mean(dPower(dtf, i))).(1:3)
        if any(isnan.(μTheo))
            return Inf
        end
        sum((μTheo .- μ[1:3]).^2)
    end
    par = (i -> params(d)[i]).(1:3)
    return optimize(vars -> tf(vars, μ), par).minimizer
end

function MTP(μ::Vector{T}, d::Geometric)::Vector{Float64} where {T <: Real}
    return [1/(μ[1] + 1)]
end

function MTP(μ::Vector{T}, d::Gumbel)::Vector{Float64} where {T <: Real}
    γ = Base.MathConstants.eulergamma
    θ = sqrt((μ[2] - μ[1]^2)*6/(π^2))
    μd = μ[1] - θ*γ
    return [μd, θ]
end

function MTP(μ::Vector{T}, d::InverseGamma)::Vector{Float64} where {T <: Real}
    θ = μ[1]*μ[2]/(μ[2] - μ[1]^2)
    α = θ/μ[1] + 1
    return [α, θ]
end

function MTP(μ::Vector{T}, d::InverseGaussian)::Vector{Float64} where {T <: Real}
    μd = μ[1]
    λ = μd^3/(μ[2] - μ[1]^2)
    return [μd, λ]
end

function MTP(μ::Vector{T}, d::Laplace)::Vector{Float64} where {T <: Real}
    μd = μ[1]
    θ = sqrt((μ[2] - μ[1]^2)/2)
    return [μd, θ]
end

function MTP(μ::Vector{T}, d::LogNormal)::Vector{Float64} where {T <: Real}
    σ = sqrt(log(μ[2]) - 2*log(μ[1]))
    μd = log(μ[2])/2 - σ^2
    return [μd, σ]
end

function MTP(μ::Vector{T}, d::Logistic)::Vector{Float64} where {T <: Real}
    μd = μ[1]
    θ = sqrt((μ[2] - μ[1]^2)*3/(π^2))
    return [μd, θ]
end

function MTP(μ::Vector{T}, d::NegativeBinomial)::Vector{Float64} where {T <: Real}
    p = μ[1]/(μ[2] - μ[1]^2)
    r = μ[1]*p/(1 - p)
    return [r, p]
end

function MTP(μ::Vector{T}, d::NoncentralChisq)::Vector{Float64} where {T <: Real}
    λ = (μ[2] - μ[1]^2)/2 - μ[1]
    ν = μ[1] - λ
    return [ν, λ]
end

function MTP(μ::Vector{T}, d::NoncentralF)::Vector{Float64} where {T <: Real}
    # No (easy) analytical solution
    function tf(θ, μ)
        dtf = try
            dtf = NoncentralF(θ[1], θ[2], θ[3])
        catch y
            if isa(y, ArgumentError) | isa(y, DomainError)
                return Inf
            end
        end
        μTheo = (i -> mean(dPower(dtf, i))).(1:3)
        if any(isnan.(μTheo))
            return Inf
        end
        sum((μTheo .- μ[1:3]).^2)
    end
    par = (i -> params(d)[i]).(1:3)
    return optimize(vars -> tf(vars, μ), par).minimizer
end

function MTP(μ::Vector{T}, d::NoncentralT)::Vector{Float64} where {T <: Real}
    # No (easy) analytical solution
    function tf(θ, μ)
        dtf = try
            dtf = NoncentralT(θ[1], θ[2])
        catch y
            if isa(y, ArgumentError) | isa(y, DomainError)
                return Inf
            end
        end
        μTheo = (i -> mean(dPower(dtf, i))).(1:2)
        if any(isnan.(μTheo))
            return Inf
        end
        sum((μTheo .- μ[1:2]).^2)
    end
    par = (i -> params(d)[i]).(1:2)
    return optimize(vars -> tf(vars, μ), par).minimizer
end

function MTP(μ::Vector{T}, d::Normal)::Vector{Float64} where {T <: Real}
    μd = μ[1]
    σ = sqrt(μ[2] - μ[1]^2)
    return [μd, σ]
end

function MTP(μ::Vector{T}, d::NormalCanon)::Vector{Float64} where {T <: Real}
    λ = 1/(μ[2] - μ[1]^2)
    η = μ[1]*λ
    return [η, λ]
end

function MTP(μ::Vector{T}, d::NormalInverseGaussian)::Vector{Float64} where {T <: Real}
    # No (easy) analytical solution
    function tf(θ, μ)
        dtf = try
            dtf = NormalInverseGaussian(θ[1], θ[2], θ[3], θ[4])
        catch y
            if isa(y, ArgumentError) | isa(y, DomainError)
                return Inf
            end
        end
        μTheo = (i -> mean(dPower(dtf, i))).(1:4)
        if any(isnan.(μTheo))
            return Inf
        end
        sum((μTheo .- μ[1:4]).^2)
    end
    par = (i -> params(d)[i]).(1:4)
    return optimize(vars -> tf(vars, μ), par).minimizer
end

function MTP(μ::Vector{T}, d::PGeneralizedGaussian)::Vector{Float64} where {T <: Real}
    # No (easy) analytical solution
    function tf(θ, μ)
        dtf = try
            dtf = PGeneralizedGaussian(θ[1], θ[2], θ[3])
        catch y
            if isa(y, ArgumentError) | isa(y, DomainError)
                return Inf
            end
        end
        μTheo = (i -> mean(dPower(dtf, i))).(1:3)
        if any(isnan.(μTheo))
            return Inf
        end
        sum((μTheo .- μ[1:3]).^2)
    end
    par = (i -> params(d)[i]).(1:3)
    return optimize(vars -> tf(vars, μ), par .+ 0.0).minimizer
end

function MTP(μ::Vector{T}, d::Pareto)::Vector{Float64} where {T <: Real}
    θ = d.θ # Fixed
    return [μ[1]/(μ[1] - θ)]
end

function MTP(μ::Vector{T}, d::Poisson)::Vector{Float64} where {T <: Real}
    return μ
end

function MTP(μ::Vector{T}, d::Rayleigh)::Vector{Float64} where {T <: Real}
    return [μ[1]/sqrt(π/2)]
end

function MTP(μ::Vector{T}, d::Skellam)::Vector{Float64} where {T <: Real}
    μd1 = ((μ[2] - μ[1]^2) + μ[1])/2
    μd2 = μd1 - μ[1]
    return [μd1, μd2]
end

# Exception since TDist symmetric around zero
function MTP(μ::Vector{T}, d::TDist)::Vector{Float64} where {T <: Real}
    return [2*μ[1]/(μ[1] - 1)]
end

function MTP(μ::Vector{T}, d::VonMises)::Vector{Float64} where {T <: Real}
    μd = μ[1]
    s = μ[2] - μ[1]^2
    function tf(θ, s)
        1 - besseli(1, θ)/besseli(0, θ) - s
    end
    return [find_zero(vars -> tf(vars, s), (0.0001, 1/d.κ), A42(), atol = 1e-05)]
end

function MTP(μ::Vector{T}, d::Weibull)::Vector{Float64} where {T <: Real}
    # No (easy) analytical solution
    function tf(θ, μ)
        dtf = try
            dtf = Weibull(θ[1], θ[2])
        catch y
            if isa(y, ArgumentError) | isa(y, DomainError)
                return Inf
            end
        end
        μTheo = (i -> mean(dPower(dtf, i))).(1:2)
        if any(isnan.(μTheo))
            return Inf
        end
        sum((μTheo .- μ[1:2]).^2)
    end
    par = (i -> params(d)[i]).(1:2)
    return optimize(vars -> tf(vars, μ), par .+ 0.0).minimizer
end
