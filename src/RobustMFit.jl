module RobustMFit

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

export dPower, rand, minimum, maximum, insupport, cdf, pdf, logpdf, quantile, skewness, kurtosis
export mean, var, std, NewDist, nParEff, MSetting, HuberSetting, TukeySetting, AndrewSetting, HampelSetting
export Huber, Tukey, Andrew, Hampel, HuberS, AndrewS, HampelS, ρ, ψ, ψder, w
export IF, MTP, MTPder, getParams, PTM, Dagum, mode, modes, CMPDist
export findCorr, findkL, findkU, updatekL, updatekU, Mfit, FInfo, AVar, RAE

end