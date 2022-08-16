module DistFit

using Distributions, Optim, QuadGK, Expectations, SpecialFunctions, ForwardDiff
using Roots, Calculus, LinearAlgebra

include("Distributions.jl")
include("nPar.jl")
include("MFunctions.jl")
include("checkConvergence.jl")
include("checkParam.jl")
include("Dagum.jl")
include("ConwayMaxwell.jl")
include("NewMFunction.jl")
include("findCorr.jl")
include("findkL.jl")
include("findkU.jl")
include("FisherInfo.jl")
include("InfluenceFunction.jl")
include("MomentsToParameter.jl")
include("MTPder.jl")
include("ParameterToMoments.jl")
include("AVar.jl")
include("RAE.jl")


end
