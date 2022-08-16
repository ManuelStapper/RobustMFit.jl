# Function to compute E(ψ(z))
# Given a distribution, ψ Function with tuning constants

function findCorr(d::T1,
                  spec::T2)::Float64 where {T1 <: Distribution{Univariate, Continuous}, T2 <: MSetting}
    μ = mean(d)
    σ = std(d)

    if !isfinite(maximum(d) - minimum(d))
        E = expectation(truncated(d, quantile(d, 0.001), quantile(d, 0.999)), n = 10000)
    else
        E = expectation(d)
    end

    E(x -> ψ((x-μ)/σ, spec))
end

function findCorr(d::T1,
                  spec::T2)::Float64 where {T1 <: Distribution{Univariate, Discrete}, T2 <: MSetting}
    lb = quantile(d, 0.0001)
    ub = quantile(d, 0.9999)
    μ = mean(d)
    σ = std(d)

    sup = lb:ub
    z = (sup .- μ)./σ
    p = pdf.(d, sup)
    return ψ.(z, spec)'p
end

# For powers
function findCorr(d::dPower,
                  spec::T)::Float64 where {T <: MSetting}
    μ = mean(d)
    σ = std(d)

    if typeof(d.d) <: Distribution{Univariate, Continuous}
        if !isfinite(maximum(d.d) - minimum(d.d))
            E = expectation(truncated(d.d, quantile(d.d, 0.001), quantile(d.d, 0.999)), n = 10000)
        else
            E = expectation(d.d)
        end

        E(x -> ψ((x^d.p-μ)/σ, spec))
    else
        sup = collect(quantile(d.d, 0.001):quantile(d.d, 0.999))
        prob = pdf.(d.d, sup)
        return sum(ψ.((sup.^d.p .- μ)./σ, spec).*prob)
    end
end

# Testing every combination
# findCorr(Exponential(5), Tukey(4))
# findCorr(Exponential(5)^2, Tukey(4))
# findCorr(Poisson(5), Tukey(4))
# findCorr(Poisson(5)^2, Tukey(10))
#
# findCorr(Exponential(5), Huber(2))
# findCorr(Exponential(5)^2, Huber(2))
# findCorr(Poisson(5), Huber(2))
# findCorr(Poisson(5)^2, Huber(3))
#
# findCorr(Exponential(5), Huber(2, ϵ = 0.2))
# findCorr(Exponential(5)^2, Huber(2, ϵ = 0.2))
# findCorr(Poisson(5), Huber(2, ϵ = 0.2))
# findCorr(Poisson(5)^2, Huber(3, ϵ = 0.2))
#
# findCorr(Exponential(5), Andrew(4))
# findCorr(Exponential(5)^2, Andrew(10))
# findCorr(Poisson(5), Andrew(4))
# findCorr(Poisson(5)^2, Andrew(10))
#
# findCorr(Exponential(5), Andrew(4, ϵ = 0.2))
# findCorr(Exponential(5)^2, Andrew(10, ϵ = 0.1))
# findCorr(Poisson(5), Andrew(4, ϵ = 0.2))
# findCorr(Poisson(5)^2, Andrew(10, ϵ = 0.2))
#
# findCorr(Exponential(5), Hampel([1, 2, 4]))
# findCorr(Exponential(5)^2, Hampel([1, 2, 4]))
# findCorr(Poisson(5), Hampel([1, 2, 4]))
# findCorr(Poisson(5)^2, Hampel([1, 2, 4]))
#
# findCorr(Exponential(5), Hampel([1, 2, 4], ϵ = 0.2))
# findCorr(Exponential(5)^2, Hampel([1, 2, 4], ϵ = 0.2))
# findCorr(Poisson(5), Hampel([1, 2, 4], ϵ = 0.2))
# findCorr(Poisson(5)^2, Hampel([1, 2, 4], ϵ = 0.2))
