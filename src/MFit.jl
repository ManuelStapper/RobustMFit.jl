module MFit

using Distributions, Optim, QuadGK, Expectations, SpecialFunctions, ForwardDiff
using Roots, Calculus, LinearAlgebra, Random

### General functions
include("Distributions.jl")
include("nPar.jl")
include("MFunctions.jl")
include("InfluenceFunction.jl")
include("checkConvergence.jl")
include("checkParam.jl")

### Parameter functions
include("MomentsToParameter.jl")
include("MTPder.jl")
include("dmu.jl")
include("getParameters.jl")
include("ParameterToMoments.jl")

### Additional Distributions/MFunctions
include("Dagum.jl")
include("CMPDist.jl")
include("NewMFunction.jl")

### Bias corrections
include("findCorr.jl")
include("findkL.jl")
include("findkU.jl")

### Estimation functions
include("RhoMom.jl")
include("RhoPar.jl")
include("PsiMom.jl")
include("PsiPar.jl")
include("wMom.jl")
include("PsiMomC.jl")

include("fit.jl")

### Inference
include("FisherInfo.jl")
include("AVar.jl")
include("RAE.jl")

end