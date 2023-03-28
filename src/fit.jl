# Function for estimation

"""
    Mfit(x::Vector{Real}, d::UnivariateDistribution, spec::MSetting;
         type::Union{Symbol, String} = :ψ, MM::Bool = true, biasCorr = :L)
    Mfit(x::Vector{Real}, d::UnivariateDistribution, spec::Vector{MSetting};
         type::Union{Symbol, String} = :ψ, MM::Bool = true, biasCorr = :L)

Fitting a univariate distribution `d` to sample `x` using M-estimation.
If `d` has multiple parameters to be estimated, different specifications can be provided. If only
one specification is added, it is used for all estimation equations.

`type` gives the type of estimation (:ρ, :ψ or :w) for estimation based on the loss function, its derivative
or the weight function respectively.

Setting `MM` to `true` carries out estimations of the raw moments and then translates estimates to parameters.
If `MM = false`, the parameters are estiamted directly. Then, only the ψ-type estimation can be performed.

`biasCorr` specifies the type of bias correction. `:L` indicates updating the lower tuning constant,
`:U` indicates updating the upper tuning constant, `biasCorr = :C` carries out estimation based
on correction terms and `:N` does not include a bias correction.

Default is ψ-type estimation based on moments with updating the lower tuning constant.

# Example
```julia
d = Poisson(10)
x = rand(d, 200)
spec = Huber(1.5)
Mfit(x, d, spec)

d = Normal()
x = rand(d, 200)
spec = [Huber(1.5), Huber(2.5)]
Mfit(x, d, spec)
```
"""
function Mfit(x::Vector{T1},
    d::T2,
    spec::T3;
    type::Union{Symbol,String}=:ψ,
    MM::Bool=true,
    biasCorr::Union{Symbol,String}=:L,
    maxIter::Int64 = 1000,
    conv::Float64=1e-05) where {T1<:Real,T2<:UnivariateDistribution,T3<:MSetting}
    if nParEff(d) > 1
        spec = fill(spec, nParEff(d))
    end
    fname = Symbol(type, ifelse(MM, :Mom, :Par), biasCorr)
    return eval(Expr(:call, fname, d, x, spec, maxIter, conv))
end


function Mfit(x::Vector{T1},
    d::T2,
    spec::Vector{T3};
    type::Union{Symbol,String}=:ψ,
    MM::Bool=true,
    biasCorr::Union{Symbol,String}=:L,
    maxIter::Int64 = 1000,
    conv::Float64=1e-05) where {T1<:Real,T2<:UnivariateDistribution,T3<:MSetting}
    fname = Symbol(type, ifelse(MM, :Mom, :Par), biasCorr)
    if length(spec) < nParEff(d)
        spec = fill(spec, nParEff(d))
    end
    return eval(Expr(:call, fname, d, x, spec, maxIter, conv))
end
