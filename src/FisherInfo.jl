# Functions to calculate the Fisher Information Matrix

# Fallback function for general case:
function FInfo(d::T) where {T <: Distribution{Univariate, Continuous}}
    # Parameter names
    npar = nPar(d)
    # Parameter vector
    θ = (i -> Distributions.params(d)[i]).(1:npar)

    function ll(θ, x, d)
        dll = NewDist(d, θ)
        logpdf(dll, x)
    end

    if isfinite(minimum(d) - maximum(d))
        E = expectation(d, n = 1000)
    else
        E = expectation(truncated(d, quantile(d, 0.0001), quantile(d, 0.9999)), n = 1000)
    end

    E(x -> Calculus.hessian(θ -> -ll(θ, x, d), θ))
end

function FInfo(d::T) where {T <: Distribution{Univariate, Discrete}}
    # Parameter names
    npar = nPar(d)
    # Parameter vector
    θ = (i -> Distributions.params(d)[i]).(1:npar)

    function ll(θ, x, d)
        dll = NewDist(d, θ)
        logpdf(dll, x)
    end

    l = quantile(d, 0.0001)
    u = quantile(d, 0.9999)
    sup = collect(l:u)
    H = x -> Calculus.hessian(θ -> -ll(θ, x, d), θ)
    sum((x -> H(x).*pdf(d, x)).(sup))
end

# Specific functions

function FInfo(d::Beta{Float64})
    E = expectation(d, n = 10000)
    e1 = E(x -> log(x)^2)
    e2 = E(x -> log(x))
    e3 = E(x -> log(1 - x)^2)
    e4 = E(x -> log(1 - x))
    e5 = E(x -> log(x)*log(1 - x))

    [e1 - e2^2 e5 - e2*e4; e5 - e2*e4 e3 - e4^2]
end # checked

function FInfo(d::BetaBinomial{Float64})
    n = d.n
    α = d.α
    β = d.β

    sup = 0:n
    p = pdf.(d, sup)

    d1 = (x -> digamma(x+α) + digamma(α + β) - digamma(n + α + β) - digamma(α)).(sup)
    d2 = (x -> digamma(n-x+β) + digamma(α + β) - digamma(n + α + β) - digamma(β)).(sup)

    m1 = p'*(d1.^2)
    m2 = p'*(d1.*d2)
    m3 = p'*(d2.^2)
    [m1 m2; m2 m3]
end # checked

function FInfo(d::BetaPrime{Float64})
    E = expectation(truncated(d, 0, quantile(d, 0.999)), n = 10000)
    e1 = E(x -> (log(x) - log(1 + x))^2)
    e2 = E(x -> log(x) - log(1 + x))
    e3 = E(x -> log(1 + x)^2)
    e4 = E(x -> -log(1 + x))
    e5 = E(x -> -(log(x) - log(1 + x))*log(1 + x))

    [e1 - e2^2 e5 - e2*e4; e5 - e2*e4 e3 - e4^2]
end # checked

function FInfo(d::Binomial{Float64})
    n = d.n
    p = d.p
    n/(p*(1 - p))
end # checked

function FInfo(d::Chi{Float64})
    E = expectation(truncated(d, 0, quantile(d, 0.999)), n = 10000)
    E(x -> log(x)^2) - E(x -> log(x))^2
end # checked

function FInfo(d::Chisq{Float64})
    E = expectation(truncated(d, 0, quantile(d, 0.999)), n = 10000)
    (E(x -> log(x)^2) - E(x -> log(x))^2)/4
end # checked

function FInfo(d::Erlang{Float64})
    d.α/d.θ^2
end # checked

function FInfo(d::Exponential{Float64})
    1/d.θ^2
end # checked

function FInfo(d::FDist{Float64})
    ν1 = d.ν1
    ν2 = d.ν2
    test = (x -> Calculus.gradient(θ -> logpdf(FDist(θ[1], θ[2]), x))([ν1, ν2]))
    E = expectation(truncated(d, quantile(d, 0.001), quantile(d, 0.999)), n = 10000)
    S = E(x -> test(x)*test(x)')
    S
end # Still a Problem here?

function FInfo(d::Gamma{Float64})
    α = d.α
    θ = d.θ
    E = expectation(truncated(d, 0, quantile(d, 0.999)), n = 10000)
    f1 = x -> log(x) - log(θ) - digamma(α)
    f2 = x -> x/θ^2 - α/θ
    return E(x -> [f1(x), f2(x)]*[f1(x), f2(x)]')
end # checked

# Slightly inaccurate?
# Problem: Support depends on parameters
# function FInfo(d::GeneralizedExtremeValue{Float64})
#     μ = d.μ
#     σ = d.σ
#     ξ = d.ξ
#
#     if ξ != 0
#         t = x -> (1 + ξ*((x-μ)/σ))^(-1/ξ)
#         dt1 = x -> ((1 + ξ*((x-μ)/σ))^(-(1/ξ + 1)))/σ
#     else
#         t = x -> exp(-(x-μ)/σ)
#         dt1 = x -> exp(-(x-μ)/σ)/σ
#     end
#     dt2 = x -> (x-μ)/σ*dt1(x)
#     dt3 = x -> t(x)*(log(1 + ξ*((x-μ)/σ))/ξ^2 - (x-μ)/(ξ*σ*(1 + ξ*((x-μ)/σ))))
#
#     dlt1 = x -> dt1(x)/t(x)
#     dlt2 = x -> dt2(x)/t(x)
#     dlt3 = x -> dt3(x)/t(x)
#
#     E = expectation(truncated(d, quantile(d, 0.001), quantile(d, 0.999)))
#     dlf1 = x -> (ξ+1)*dlt1(x) - dt1(x)
#     dlf2 = x -> -1/σ + (ξ + 1)*dlt2(x) - dt2(x)
#     dlf3 = x -> (ξ+1)*dlt3(x) + log(t(x)) - dt3(x)
#     dlf = x -> [dlf1(x), dlf2(x), dlf3(x)]
#
#     return E(x -> dlf(x)*dlf(x)')
# end

# Slightly inaccurate?
# Problem: Support depends on parameters
# function FInfo(d::GeneralizedPareto{Float64})
#     μ = d.μ
#     σ = d.σ
#     ξ = d.ξ
#
#     z(x) = (x - μ)/σ
#
#     dlf1 = x -> (1 + ξ)/(1 - ξ*z(x)*σ)
#     dlf2 = x -> (1/ξ + 1)*z(x)/((1 + ξ*z(x))*σ)
#     dlf3 = x -> (ξ*z(x) + z(x))/(ξ^2*z(x)) - log(ξ*z(x) + 1)/(ξ^2)
#     dlf = x -> [dlf1(x), dlf2(x), dlf3(x)]
#
#     E = expectation(truncated(d, quantile(d, 0.001), quantile(d, 0.999)))
#
#     E(x -> dlf(x)*dlf(x)')
# end

function FInfo(d::Geometric{Float64})
    1/(d.p^2*(1 - d.p))
end # checked

function FInfo(d::Gumbel{Float64})
    μ = d.μ
    θ = d.θ

    test = (x -> Calculus.gradient(θ2 -> logpdf(Gumbel(θ2[1], θ2[2]), x))([μ, θ]))
    E = expectation(truncated(d, quantile(d, 0.001), quantile(d, 0.999)), n = 10000)
    S = E(x -> test(x)*test(x)')
    S
end # checked


function FInfo(d::InverseGamma{Float64})
    α = d.invd.α
    θ = d.θ
    E = expectation(truncated(d, 0, quantile(d, 0.999)), n = 10000)
    dlf1 = x -> log(θ) - digamma(α) - log(x)
    dlf2 = x -> α/θ - 1/x
    dlf = x -> [dlf1(x), dlf2(x)]
    return E(x -> dlf(x)*dlf(x)')
end # checked

function FInfo(d::InverseGaussian{Float64})
    μ = d.μ
    λ = d.λ

    dlf1 = x -> -λ*(x - μ)/(μ^3)
    dlf2 = x -> 1/(2*λ) - (x-μ)^2/(2*μ^2*x)
    dlf = x -> [dlf1(x), dlf2(x)]

    E = expectation(truncated(d, 0, quantile(d, 0.999)), n = 10000)
    E(x -> dlf(x)*dlf(x)')
end # checked

function FInfo(d::Laplace{Float64})
    μ = d.μ
    θ = d.θ

    dlf1 = x -> sign(x-μ)/θ
    dlf2 = x -> -1/θ + sign(x - μ)*(x-μ)/θ^2
    dlf = x -> [dlf1(x), dlf2(x)]

    E = expectation(truncated(d, quantile(d, 0.001), quantile(d, 0.999)), n = 10000)
    E(x -> dlf(x)*dlf(x)')
end # checked

function FInfo(d::LogNormal{Float64})
    [1/d.σ^2 0; 0 2/d.σ^2]
end

function FInfo(d::Logistic{Float64})
    μ = d.μ
    θ = d.θ

    η = x -> exp(-(x-μ)/θ)
    dη1 = x -> η(x)/θ
    dη2 = x -> η(x)*(x-μ)/θ^2
    dlf1 = x -> (1/η(x) - 2/(1 + η(x)))*dη1(x)
    dlf2 = x -> (1/η(x) - 2/(1 + η(x)))*dη2(x) - 1/θ
    dlf = x -> [dlf1(x), dlf2(x)]
    E = expectation(truncated(d, quantile(d, 0.001), quantile(d, 0.999)), n = 10000)

    E(x -> dlf(x)*dlf(x)')
end # checked

function FInfo(d::NegativeBinomial{Float64})
    r = d.r
    p = d.p
    sup = collect(0:quantile(d, 0.999))

    dlf1 = x -> digamma(x+r) - digamma(r) + log(p)
    dlf2 = x -> r/p  - x/(1 - p)
    dlf = x -> [dlf1(x), dlf2(x)]
    sum((x -> pdf(d, x) .* dlf(x)*dlf(x)').(sup))
end # checked

function FInfo(d::Normal{Float64})
    σ = d.σ
    [1/σ^2 0; 0 2/σ^2]
end # checked

function FInfo(d::NormalCanon{Float64})
    μ = η/λ
    σ = sqrt(1/λ)
    H'*inv(FInfo(Normal(μ, σ)))*H
end # checked

function FInfo(d::NormalInverseGaussian{Float64})
    μ = d.μ
    α = d.α
    β = d.β
    δ = d.δ
    γ = sqrt(α^2 - β^2)
    η = x -> sqrt(δ^2 + (x - μ)^2)

    k0 = x -> besselk(0, x)
    k1 = x -> besselk(1, x)
    k2 = x -> besselk(2, x)

    dlf1 = x -> (x - μ)/η(x)^2 - β + α*(k0(α*η(x)) + k2(α*η(x)))/(2*k1(α*η(x)))*(x-μ)/η(x)
    dlf2 = x -> 1/α + α*δ/γ - η(x)*(k0(α*η(x)) + k2(α*η(x)))/(2*k1(α*η(x)))
    dlf3 = x -> -β*δ/γ + x - μ
    dlf4 = x -> 1/δ - δ/η(x)^2 + γ - α*(k0(α*η(x)) + k2(α*η(x)))/(2*k1(α*η(x)))*δ/η(x)

    dlf = x -> [dlf1(x), dlf2(x), dlf3(x), dlf4(x)]

    E = expectation(truncated(d, quantile(d, 0.001), quantile(d, 0.999)), n = 10000)
    E(x -> dlf(x)*dlf(x)')
end # Should be correct

function FInfo(d::PGeneralizedGaussian)
    μ = d.μ + 0.0
    α = d.α + 0.0
    p = d.p + 0.0

    dlf1 = x -> p*α^(-p)*(x-μ)*abs(x-μ)^p
    dlf2 = x -> -1/α - p*α^(-(p+1))*abs(x-μ)^p
    dlf3 = x -> 1/p + digamma(1/p)/p^2 - α^(-p)*abs(x-μ)^p*log(abs(x-μ)/α)

    dlf = x -> [dlf1(x), dlf2(x), dlf3(x)]

    E = expectation(truncated(d, quantile(d, 0.001), quantile(d, 0.999)), n = 10000)
    E(x -> dlf(x)*dlf(x)')
end # ML Estimation produces weird estimates

function FInfo(d::Pareto{Float64})
    1/d.α^2
end # checked

function FInfo(d::Poisson{Float64})
    1/d.λ
end # checked

function FInfo(d::Rayleigh{Float64})
    # central moments
    m2c = σ^2*(4-π)/2
    m3c = (2*sqrt(π)*(π-3)/((4 - π)^(3/2)))*(m2c)^(3/2)
    m4c = (-(6*π^2 - 24*π + 16)/((4 - π)^2) + 3)*m2c^2

    # raw moments
    m1r = σ*sqrt(π/2)
    m2r = m2c + m1r^2
    m3r = m3c + 3*m1r*m2r - 2*m1r^3
    m4r = m4c + 4*m1r*m3r - 6*m1r^2*m2r + 3*m1r^4

    (m4r - m2r^2)/σ^6
end # checked

function FInfo(d::Skellam{Float64})
    μ1 = d.μ1
    μ2 = d.μ2

    mm = 2*sqrt(μ1*μ2)

    l = -quantile(Poisson(μ2), 0.99999)
    u = quantile(Poisson(μ1), 0.99999)
    sup = collect(l:u)
    p = pdf.(d, sup)

    besselDer = x -> (besseli(x-1, mm) + besseli(x+1, mm))/(2*besseli(x, mm))
    dlf1 = x -> -1 + x/(2*μ1) + besselDer(x)*μ2/(sqrt(μ1*μ2))
    dlf2 = x -> -1 - x/(2*μ2) + besselDer(x)*μ1/(sqrt(μ1*μ2))

    dlf = x -> [dlf1(x), dlf2(x)]

    sum((x -> pdf(d, x) .* dlf(x)*dlf(x)').(sup))
end # checked

function FInfo(d::TDist{Float64})
    ν = d.ν

    dlf = x -> digamma((ν+1)/2)/2 - 1/(2*ν) - digamma(ν/2)/2 - log(1 + x^2/ν)/2 + ((ν+1)/2)*x^2/(ν*(ν + x^2))
    E = expectation(truncated(d, quantile(d, 0.001), quantile(d, 0.999)), n = 10000)
    E(x -> dlf(x)^2)
end # checked

function FInfo(d::VonMises{Float64})
    μ = d.μ
    κ = d.κ

    dlf1 = x -> κ*sin(x - μ)
    dlf2 = x -> cos(x - μ) - besseli(1, κ)/besseli(0, κ)
    dlf = x -> [dlf1(x), dlf2(x)]

    E = expectation(d, n = 10000)
    E(x -> dlf(x)*dlf(x)')
end # checked


function FInfo(d::Weibull{Float64})
    α = d.α
    θ = d.θ

    dlf1 = x -> 1/α + log(x) - log(θ) - (x/θ)^α*log(x/θ)
    dlf2 = x -> -α/θ*(1 - (x/θ)^α)
    dlf = x -> [dlf1(x), dlf2(x)]

    E = expectation(truncated(d, 0, quantile(d, 0.999)), n = 10000)
    E(x -> dlf(x)*dlf(x)')
end # checked
