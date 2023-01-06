using RobustMFit
using Test

@testset "RobustMFit.jl" begin
    using Random, Distributions

    d = Poisson(10)
    d2 = dPower(d, 2)
    d3 = dPower(d, 3)
    d4 = dPower(d, 4)
    d5 = dPower(d, 5)
    rand(d2)
    rand(d2, 10)
    rand(d2, (10, 10))
    minimum(d2)
    maximum(d2)
    insupport(d2, 9)
    cdf(d2, 10)
    pdf(d2, 9)
    pdf(d3, 125)
    logpdf(d2, 9)
    quantile(d2, 0.4)
    quantile(d3, 0.4)
    skewness(d2)
    kurtosis(d2)
    mean(d2)
    var(d2)
    mean(d3)
    mean(d4)
    mean(d5)
    std(d2)
    NewDist(d, [11])
    nParEff(d)

    d = Normal()
    d2 = dPower(d, 2)
    d3 = dPower(d, 3)
    d4 = dPower(d, 4)
    d5 = dPower(d, 5)
    rand(d2)
    rand(d2, 10)
    rand(d2, (10, 10))
    minimum(d2)
    maximum(d2)
    insupport(d2, 9)
    cdf(d2, 10)
    pdf(d2, 9)
    pdf(d3, 125)
    logpdf(d2, 9)
    quantile(d2, 0.4)
    quantile(d3, 0.4)
    skewness(d2)
    kurtosis(d2)
    mean(d2)
    var(d2)
    mean(d3)
    mean(d4)
    mean(d5)
    std(d2)
    NewDist(d, [1.0, 1])
    nParEff(d)

    spec = [Huber(1.5), Huber(1.5, 2.5), HuberS(1.5, 2.5, 0.1), Huber(1.5, ϵ=0.1),
        Tukey(4.5), Tukey(4.5, 5.5),
        Andrew(4.5), Andrew(4.5, 6), AndrewS(4.5, 6, 0.1), Andrew(4.5, ϵ=0.1),
        Hampel([1, 2, 5]), Hampel([1, 2, 5], [1, 3, 9]), Hampel([1, 2, 5], ϵ=0.1)]

    for i = 1:length(spec)
        ρ(0, spec[i])
        ψ(0, spec[i])
        ψder(0, spec[i])
        w(0, spec[i])
    end

    d = Poisson(10)
    IF(d, Huber(1.5))(9)
    IF(dPower(d, 2), Huber(1.5))(9^2)

    d = Normal()
    IF(d, Huber(1.5))(1)
    IF(dPower(d, 2), Huber(1.5))(2)


    # All distributions?
    AllDists = [Beta(),
        BetaBinomial(1, 2, 3),
        BetaPrime(2, 3),
        Binomial(),
        Chi(1),
        Chisq(1),
        Erlang(),
        Exponential(),
        FDist(5, 5),
        Gamma(),
        GeneralizedExtremeValue(1, 2, 0.1),
        GeneralizedPareto(0, 1, 0.1),
        Geometric(),
        Gumbel(),
        InverseGamma(),
        InverseGaussian(),
        Laplace(),
        LogNormal(),
        Logistic(),
        NegativeBinomial(),
        NoncentralChisq(1, 1),
        NoncentralT(1, 1),
        Normal(),
        NormalCanon(),
        # NormalInverseGaussian(0, 1, 0, 1),
        PGeneralizedGaussian(),
        Pareto(),
        Poisson(),
        Rayleigh(),
        Skellam(),
        VonMises(),
        Weibull()]

    for i = 1:31
        m = PTM(AllDists[i])
        MTP(m, AllDists[i])
        MTPder(m, AllDists[i])
        RobustMFit.dμ(AllDists[i])
        getParams(AllDists[i])
    end

    for i = 1:31
        if i != 22
            FInfo(AllDists[i])
        end
    end

    d = Dagum(5, 1, 1)
    mean(d)
    var(d)
    skewness(d)
    kurtosis(d)
    params(d)
    mode(d)
    modes(d)
    pdf(d, 1)
    cdf(d, 1)
    median(d)
    quantile(d, 0.4)
    rand(d)
    rand(d, 10)
    rand(d, (10, 10))
    pdf.(d, [1, 2])

    d = CMPDist(5, 0.7)
    mean(d)
    var(d)
    skewness(d)
    kurtosis(d)
    params(d)
    mode(d)
    modes(d)
    pdf(d, 1)
    cdf(d, 1)
    median(d)
    quantile(d, 0.4)
    rand(d)
    rand(d, 10)
    rand(d, (10, 10))
    pdf.(d, [1, 2])

    RobustMFit.MyType(1.5)
    RobustMFit.MyType(1.5, 2.5)
    RobustMFit.MyTypeS(1.5, 2.5, 0.1)
    spec2 = RobustMFit.MyType(1.5, ϵ=0.1)
    ρ(0, spec2)
    ψ(0, spec2)
    ψder(0, spec2)
    w(1, spec2)

    # Testing every combination
    d = Exponential(5)
    findCorr(d, Tukey(4))
    findCorr(dPower(d, 2), Tukey(4))
    findCorr(d, Huber(2))
    findCorr(dPower(d, 2), Huber(2))
    findCorr(d, Huber(2, ϵ=0.2))
    findCorr(dPower(d, 2), Huber(2, ϵ=0.2))
    findCorr(d, Andrew(4))
    findCorr(dPower(d, 2), Andrew(10))
    findCorr(d, Andrew(4, ϵ=0.2))
    findCorr(dPower(d, 2), Andrew(10, ϵ=0.1))
    findCorr(d, Hampel([1, 2, 4]))
    findCorr(dPower(d, 2), Hampel([1, 2, 4]))
    findCorr(d, Hampel([1, 2, 4], ϵ=0.2))
    findCorr(dPower(d, 2), Hampel([1, 2, 4], ϵ=0.2))

    d = Poisson(5)
    findCorr(d, Tukey(4))
    findCorr(dPower(d, 2), Tukey(10))
    findCorr(d, Huber(2))
    findCorr(dPower(d, 2), Huber(3))
    findCorr(d, Huber(2, ϵ=0.2))
    findCorr(dPower(d, 2), Huber(3, ϵ=0.2))
    findCorr(d, Andrew(4))
    findCorr(dPower(d, 2), Andrew(10))
    findCorr(d, Andrew(4, ϵ=0.2))
    findCorr(dPower(d, 2), Andrew(10, ϵ=0.2))
    findCorr(d, Hampel([1, 2, 4]))
    findCorr(dPower(d, 2), Hampel([1, 2, 4]))
    findCorr(d, Hampel([1, 2, 4], ϵ=0.2))
    findCorr(dPower(d, 2), Hampel([1, 2, 4], ϵ=0.2))

    # Testing every combination
    findkL(Exponential(5), Tukey(4))
    findkL(dPower(Exponential(5), 2), Tukey(4))
    findkL(Poisson(5), Tukey(4))
    findkL(dPower(Poisson(5), 2), Tukey(10))

    findkL(Exponential(5), Huber(2))
    findkL(dPower(Exponential(5), 2), Huber(2))
    findkL(Poisson(5), Huber(2))
    findkL(dPower(Poisson(5), 2), Huber(3))

    findkL(Exponential(5), Huber(2, ϵ=0.2))
    findkL(dPower(Exponential(5), 2), Huber(2, ϵ=0.2))
    findkL(Poisson(5), Huber(2, ϵ=0.2))
    findkL(dPower(Poisson(5), 2), Huber(3, ϵ=0.2))

    findkL(Exponential(5), Andrew(4))
    findkL(dPower(Exponential(5), 2), Andrew(10))
    findkL(Poisson(5), Andrew(4))
    findkL(dPower(Poisson(5), 2), Andrew(10))

    findkL(Exponential(5), Andrew(4, ϵ=0.2))
    findkL(dPower(Exponential(5), 2), Andrew(10, ϵ=0.1))
    findkL(Poisson(5), Andrew(4, ϵ=0.2))
    findkL(dPower(Poisson(5), 2), Andrew(10, ϵ=0.2))

    findkL(Exponential(5), Hampel([1, 2, 4]))
    findkL(dPower(Exponential(5), 2), Hampel([1, 2, 4]))
    findkL(Poisson(5), Hampel([1, 2, 4]))
    findkL(dPower(Poisson(5), 2), Hampel([1, 2, 4]))

    findkL(Exponential(5), Hampel([1, 2, 4], ϵ=0.2))
    findkL(dPower(Exponential(5), 2), Hampel([1, 2, 4], ϵ=0.2))
    findkL(Poisson(5), Hampel([1, 2, 4], ϵ=0.2))
    findkL(dPower(Poisson(5), 2), Hampel([1, 2, 4], ϵ=0.2))

    # Testing every combination
    findkU(Exponential(5), Tukey(2))
    findkU(dPower(Exponential(5), 2), Tukey(1))
    findkU(Poisson(5), Tukey(2))
    findkU(dPower(Poisson(5), 2), Tukey(2))

    findkU(Exponential(5), Huber(0.5))
    findkU(dPower(Exponential(5), 2), Huber(0.25))
    findkU(Poisson(5), Huber(0.5))
    findkU(dPower(Poisson(5), 2), Huber(0.5))

    findkU(Exponential(5), Huber(0.5, ϵ=0.2))
    findkU(dPower(Exponential(5), 2), Huber(0.5, ϵ=0.2))
    findkU(Poisson(5), Huber(0.5, ϵ=0.2))
    findkU(dPower(Poisson(5), 2), Huber(0.5, ϵ=0.2))

    findkU(Exponential(5), Andrew(2))
    findkU(dPower(Exponential(5), 2), Andrew(2))
    findkU(Poisson(5), Andrew(2))
    findkU(dPower(Poisson(5), 2), Andrew(2))

    findkU(Exponential(5), Andrew(2, ϵ=0.2))
    findkU(dPower(Exponential(5), 2), Andrew(2, ϵ=0.1))
    findkU(Poisson(5), Andrew(2, ϵ=0.2))
    findkU(dPower(Poisson(5), 2), Andrew(2, ϵ=0.2))

    findkU(Exponential(5), Hampel([0.2, 0.5, 0.8]))
    findkU(dPower(Exponential(5), 2), Hampel([0.2, 0.5, 0.8]))
    findkU(Poisson(5), Hampel([0.2, 0.5, 0.8]))
    findkU(dPower(Poisson(5), 2), Hampel([0.2, 0.5, 0.8]))

    findkU(Exponential(5), Hampel([0.2, 0.5, 0.8], ϵ=0.2))
    findkU(dPower(Exponential(5), 2), Hampel([0.2, 0.5, 0.8], ϵ=0.2))
    findkU(Poisson(5), Hampel([0.2, 0.5, 0.8], ϵ=0.2))
    findkU(dPower(Poisson(5), 2), Hampel([0.2, 0.5, 0.8], ϵ=0.2))

    d = Poisson(10)
    Random.seed!(06012023)
    x = rand(d, 200)
    spec1 = Huber(1.5)
    spec2 = Huber(0.5)
    Mfit(x, d, spec1, type=:ρ, MM=true, biasCorr=:L)
    Mfit(x, d, spec2, type=:ρ, MM=true, biasCorr=:U)
    Mfit(x, d, spec1, type=:ρ, MM=true, biasCorr=:N)

    Mfit(x, d, spec1, type=:ρ, MM=false, biasCorr=:L)
    Mfit(x, d, spec2, type=:ρ, MM=false, biasCorr=:U)
    Mfit(x, d, spec1, type=:ρ, MM=false, biasCorr=:N)

    Mfit(x, d, spec1, type=:ψ, MM=true, biasCorr=:L)
    Mfit(x, d, spec2, type=:ψ, MM=true, biasCorr=:U)
    Mfit(x, d, spec1, type=:ψ, MM=true, biasCorr=:N)
    Mfit(x, d, spec1, type=:ψ, MM=true, biasCorr=:C)

    Mfit(x, d, spec1, type=:ψ, MM=false, biasCorr=:L)
    Mfit(x, d, spec2, type=:ψ, MM=false, biasCorr=:U)
    Mfit(x, d, spec1, type=:ψ, MM=false, biasCorr=:N)

    Mfit(x, d, spec1, type=:w, MM=true, biasCorr=:L)
    Mfit(x, d, spec2, type=:w, MM=true, biasCorr=:U)
    Mfit(x, d, spec1, type=:w, MM=true, biasCorr=:N)

    # Tests for asymptotic variance:

    d = Poisson(10)
    Random.seed!(06012023)
    x = rand(d, 200)
    inv(FInfo(d))
    AVar(d, Huber(1.5), :L)
    AVar(d, Huber(1.5), :U)
    AVar(d, Huber(1.5), :N)
    AVar(d, Huber(1.5), :C)

    AVar(x, d, Huber(1.5), :L)
    AVar(x, d, Huber(1.5), :U)
    AVar(x, d, Huber(1.5), :N)
    AVar(x, d, Huber(1.5), :C)

    d = NegativeBinomial(3.2, 0.2)
    x = rand(d, 1000)
    inv(FInfo(d))
    AVar(d, [Huber(1.5), Huber(2.5)], :L)
    AVar(d, [Huber(0.5), Huber(0.5)], :U)
    AVar(d, [Huber(1.5), Huber(2.5)], :N)
    AVar(d, [Huber(1.5), Huber(2.5)], :C)

    AVar(x, d, [Huber(1.5), Huber(2.5)], :L)
    AVar(x, d, [Huber(0.5), Huber(0.5)], :U)
    AVar(x, d, [Huber(1.5), Huber(2.5)], :N)
    AVar(x, d, [Huber(1.5), Huber(2.5)], :C)

    d = Exponential(10)
    x = rand(d, 200)
    inv(FInfo(d))
    AVar(d, Huber(1.5), :L, n=100)
    AVar(d, Huber(1.5), :L, n=500)
    AVar(d, Huber(0.5), :U)
    AVar(d, Huber(1.5), :N)
    AVar(d, Huber(1.5), :C)

    AVar(x, d, Huber(1.5), :L)
    AVar(x, d, Huber(0.5), :U)
    AVar(x, d, Huber(1.5), :N)
    AVar(x, d, Huber(1.5), :C)

    d = Normal(3.2, 2)
    x = rand(d, 1000)
    inv(FInfo(d))
    AVar(d, [Huber(1.5), Huber(2.5)], :L)
    AVar(d, [Huber(0.5), Huber(0.5)], :U)
    AVar(d, [Huber(1.5), Huber(2.5)], :N)
    AVar(d, [Huber(1.5), Huber(2.5)], :C)

    AVar(x, d, [Huber(1.5), Huber(2.5)], :L)
    AVar(x, d, [Huber(0.5), Huber(0.5)], :U)
    AVar(x, d, [Huber(1.5), Huber(2.5)], :N)
    AVar(x, d, [Huber(1.5), Huber(2.5)], :C)

    d = Binomial(10, 0.2)
    AVar(d, Huber(1.5), :L)
    d = Erlang(3, 0.2)
    AVar(d, Huber(1.5), :L)


    # Tests for RAE:

    d = Poisson(10)
    Random.seed!(06012023)
    x = rand(d, 200)
    inv(FInfo(d))
    RAE(d, Huber(1.5), :L)
    RAE(d, Huber(1.5), :U)
    RAE(d, Huber(1.5), :N)
    RAE(d, Huber(1.5), :C)

    RAE(x, d, Huber(1.5), :L)
    RAE(x, d, Huber(1.5), :U)
    RAE(x, d, Huber(1.5), :N)
    RAE(x, d, Huber(1.5), :C)

    d = NegativeBinomial(3.2, 0.2)
    x = rand(d, 1000)
    inv(FInfo(d))
    RAE(d, [Huber(1.5), Huber(2.5)], :L)
    RAE(d, [Huber(0.5), Huber(0.5)], :U)
    RAE(d, [Huber(1.5), Huber(2.5)], :N)
    RAE(d, [Huber(1.5), Huber(2.5)], :C)

    RAE(x, d, [Huber(1.5), Huber(2.5)], :L)
    RAE(x, d, [Huber(0.5), Huber(0.5)], :U)
    RAE(x, d, [Huber(1.5), Huber(2.5)], :N)
    RAE(x, d, [Huber(1.5), Huber(2.5)], :C)

    d = Exponential(10)
    x = rand(d, 200)
    inv(FInfo(d))
    RAE(d, Huber(1.5), :L, n=100)
    RAE(d, Huber(1.5), :L, n=500)
    RAE(d, Huber(0.5), :U)
    RAE(d, Huber(1.5), :N)
    RAE(d, Huber(1.5), :C)

    RAE(x, d, Huber(1.5), :L)
    RAE(x, d, Huber(0.5), :U)
    RAE(x, d, Huber(1.5), :N)
    RAE(x, d, Huber(1.5), :C)

    d = Normal(3.2, 2)
    x = rand(d, 1000)
    inv(FInfo(d))
    RAE(d, [Huber(1.5), Huber(2.5)], :L)
    RAE(d, [Huber(0.5), Huber(0.5)], :U)
    RAE(d, [Huber(1.5), Huber(2.5)], :N)
    RAE(d, [Huber(1.5), Huber(2.5)], :C)

    RAE(x, d, [Huber(1.5), Huber(2.5)], :L)
    RAE(x, d, [Huber(0.5), Huber(0.5)], :U)
    RAE(x, d, [Huber(1.5), Huber(2.5)], :N)
    RAE(x, d, [Huber(1.5), Huber(2.5)], :C)

    d = Binomial(10, 0.2)
    RAE(d, Huber(1.5), :L)
    d = Erlang(3, 0.2)
    RAE(d, Huber(1.5), :L)
end
