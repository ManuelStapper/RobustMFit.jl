# Implement ρ, ψ, derivative of ψ and weight functions
# Huber, Tukey, Andrew and Hampel with symmetric tuning constants and with
# asymmetric tuning constants
# Functions with no continuois derivative of ψ are implemented as well
# in a smoothed version

# Plan:
# - Structs for settings
# - Constructors for settings
# - ρ, ψ, ψder and weight functions

################################################################
### Step 1:                                                  ###
### Settings with tuning constants and smoothing coefficient ###
################################################################

abstract type MSetting end
abstract type HuberSetting <: MSetting end
abstract type TukeySetting <: MSetting end
abstract type AndrewSetting <: MSetting end
abstract type HampelSetting <: MSetting end

mutable struct Huber <: HuberSetting
    kL::Float64
    kU::Float64
end

mutable struct Tukey <: TukeySetting
    kL::Float64
    kU::Float64
end

mutable struct Andrew <: AndrewSetting
    kL::Float64
    kU::Float64
end

mutable struct Hampel <: HampelSetting
    kL::Vector{Float64}
    kU::Vector{Float64}
end

mutable struct HuberS <: HuberSetting
    kL::Float64
    kU::Float64
    ϵ::Float64
end

mutable struct AndrewS <: AndrewSetting
    kL::Float64
    kU::Float64
    ϵ::Float64
end

mutable struct HampelS <: HampelSetting
    kL::Vector{Float64}
    kU::Vector{Float64}
    ϵ::Float64
end

# Include some basic function for vector-wise evaluation
import Base.length
function length(x::T) where {T <: MSetting}
    1
end

import Base.iterate
function iterate(x::T) where {T <: MSetting}
    (x, nothing)
end
function iterate(x::T, foo::Nothing) where {T <: MSetting}

end

import Base.copy
function copy(x::T) where {T <: MSetting}
    if :ϵ in propertynames(x)
        return typeof(x)(x.kL, x.kU, x.ϵ)
    else
        return typeof(x)(x.kL, x.kU)
    end
end

#################################################################
### Step 2:                                                   ###
### Constructors for settings. ϵ as keyword argument to avoid ###
### ambiguity                                                 ###
#################################################################

function Huber(kL::T1, kU::T2; ϵ::T3 = 0.0) where {T1, T2, T3 <: Real}
    if kL <= 0
        error("Invalid lower tuning constant")
    end
    if kU <= 0
        error("Invalid upper tuning constant")
    end
    if ϵ < 0
        error("Invalid smoothing parameter ϵ")
    end
    if ϵ == 0
        return Huber(Float64(kL), Float64(kU))
    else
        return HuberS(Float64(kL), Float64(kU), Float64(ϵ))
    end
end

function Huber(k::T1; ϵ::T2 = 0.0) where {T1, T2 <: Real}
    if k <= 0
        error("Invalid tuning constant")
    end
    if ϵ < 0
        error("Invalid smoothing parameter ϵ")
    end
    if ϵ == 0
        return Huber(Float64(k), Float64(k))
    else
        return HuberS(Float64(k), Float64(k), Float64(ϵ))
    end
end

function Tukey(kL::T1, kU::T2) where {T1, T2 <: Real}
    if kL <= 0
        error("Invalid lower tuning constant")
    end
    if kU <= 0
        error("Invalid upper tuning constant")
    end
    Tukey(Float64(kL), Float64(kU))
end

function Tukey(k::T1) where {T1 <: Real}
    if k <= 0
        error("Invalid tuning constant")
    end
    Tukey(Float64(k), Float64(k))
end

function Andrew(kL::T1, kU::T2; ϵ::T3 = 0.0) where {T1, T2, T3 <: Real}
    if kL <= 0
        error("Invalid lower tuning constant")
    end
    if kU <= 0
        error("Invalid upper tuning constant")
    end
    if ϵ < 0
        error("Invalid smoothing parameter ϵ")
    end
    if ϵ == 0
        return Andrew(Float64(kL), Float64(kU))
    else
        return AndrewS(Float64(kL), Float64(kU), Float64(ϵ))
    end
end

function Andrew(k::T1; ϵ::T2 = 0.0) where {T1, T2 <: Real}
    if k <= 0
        error("Invalid tuning constant")
    end
    if ϵ < 0
        error("Invalid smoothing parameter ϵ")
    end
    if ϵ == 0
        return Andrew(Float64(k), Float64(k))
    else
        return AndrewS(Float64(k), Float64(k), Float64(ϵ))
    end
end

function Hampel(kL::Vector{T1}, kU::Vector{T2}; ϵ::T3 = 0.0) where {T1, T2, T3 <: Real}
    if any(kL .<= 0)
        error("Invalid lower tuning constants")
    end
    if any(kU .<= 0)
        error("Invalid upper tuning constants")
    end
    if ϵ < 0
        error("Invalid smoothing parameter ϵ")
    end
    if ϵ == 0
        return Hampel(sort(Float64.(kL)), sort(Float64.(kU)))
    else
        return HampelS(sort(Float64.(kL)), sort(Float64.(kU)), Float64(ϵ))
    end
end

function Hampel(k::T1; ϵ::T2 = 0.0) where {T1, T2 <: Real}
    if any(k .<= 0)
        error("Invalid tuning constants")
    end
    if ϵ < 0
        error("Invalid smoothing parameter ϵ")
    end
    if ϵ == 0
        return Hampel(sort(Float64.(k)), sort(Float64.(k)))
    else
        return HampelS(sort(Float64.(k)), sort(Float64.(k)), Float64(ϵ))
    end
end


##############################################################
### Step 3:                                                ###
### ρ, ψ, weight functions and derivatives of ψ functions  ###
### Only implement the more general asymmetric case        ###
##############################################################


### Huber

function ρ(z::T1, spec::T2)::Float64 where {T1 <: Real, T2 <: HuberSetting}
    kL = spec.kL
    kU = spec.kU
    if -kL <= z <= kU
        return z^2/2
    end
    if z < -kL
        return -kL*z - kL^2/2
    end
    if z > kU
        return kU*abs(z) - kU^2/2
    end
end

function ψ(z::T1, spec::Huber)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    if -kL <= z <= kU
        return z
    end
    if z < -kL
        return -kL
    end
    if z > kU
        return kU
    end
end

function ψder(z::T1, spec::Huber)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    ifelse(-kL <= z <= kU, 1.0, 0.0)
end

function w(z::T1, spec::Huber)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    if -kL <= z <= kU
        return 1.0
    end
    if z < -kL
        return -kL/z
    end
    if z > kU
        return kU/z
    end
end

### Tukey

function ρ(z::T1, spec::Tukey)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    if -kL <= z <= 0
        return kL^2/6*(1 - (1 - z^2/kL^2)^3)
    end
    if 0 <= z <= kU
        return kU^2/6*(1 - (1 - z^2/kU^2)^3)
    end
    if z < -kL
        return kL^2/6
    end
    if z > kU
        return kU^2/6
    end
end

function ψ(z::T1, spec::Tukey)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    if -kL <= z <= 0
        return z*(1 - z^2/kL^2)^2
    end
    if 0 <= z <= kU
        return z*(1 - z^2/kU^2)^2
    end
    if z < -kL
        return 0.0
    end
    if z > kU
        return 0.0
    end
end

function ψder(z::T1, spec::Tukey)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    if -kL <= z <= 0
        return 5*z^4/kL^4 - 6*z^2/kL^2 + 1
    end
    if 0 <= z <= kU
        return 5*z^4/kU^4 - 6*z^2/kU^2 + 1
    end
    return 0.0
end

function w(z::T1, spec::Tukey)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    if -kL <= z <= 0
        return (1 - z^2/kL^2)^2
    end
    if 0 <= z <= kU
        return (1 - z^2/kU^2)^2
    end
    if z < -kL
        return 0.0
    end
    if z > kU
        return 0.0
    end
end

### Andrew

function ρ(z::T1, spec::T2)::Float64 where {T1 <: Real, T2 <: AndrewSetting}
    kL = spec.kL
    kU = spec.kU
    if -kL <= z <= 0
        return kL^2/π^2*(1 - cos(π*z/kL))
    end
    if 0 <= z <= kU
        return kU^2/π^2*(1 - cos(π*z/kU))
    end
    if z < -kL
        return 2*kL^2/π^2
    end
    if z > kU
        return 2*kU^2/π^2
    end
end

function ψ(z::T1, spec::Andrew)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    if -kL <= z <= 0
        return kL/π*sin(π*z/kL)
    end
    if 0 <= z <= kU
        return kU/π*sin(π*z/kU)
    end
    if z < -kL
        return 0.0
    end
    if z > kU
        return 0.0
    end
end

function ψder(z::T1, spec::Andrew)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    if -kL <= z <= 0
        return cos(π*z/kL)
    end
    if 0 <= z <= kU
        return cos(π*z/kU)
    end
    if z < -kL
        return 0.0
    end
    if z > kU
        return 0.0
    end
end

function w(z::T1, spec::Andrew)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    if -kL <= z <= 0
        return kL/(π*z)*sin(π*z/kL)
    end
    if 0 <= z <= kU
        return kU/(π*z)*sin(π*z/kU)
    end
    if z < -kL
        return 0.0
    end
    if z > kU
        return 0.0
    end
end

### Hampel

function ρ(z::T1, spec::T2)::Float64 where {T1 <: Real, T2 <: HampelSetting}
    kL = spec.kL
    kU = spec.kU
    if z < -kL[3]
        return kL[1]*kL[2] - kL[1]^2/2 + (kL[3] - kL[2])*kL[1]/2
    elseif z < -kL[2]
        return kL[1]*kL[2] - kL[1]^2/2 + (kL[3] - kL[2])*kL[1]/2*(1 - ((kL[3] + z)/(kL[3] - kL[2]))^2)
    elseif z < -kL[1]
        return -kL[1]*z - kL[1]^2/2
    elseif z < 0
        return z^2/2
    elseif z <= kU[1]
        return z^2/2
    elseif z <= kU[2]
        return kU[1]*z - kU[1]^2/2
    elseif z <= kU[3]
        return kU[1]*kU[2] - kU[1]^2/2 + (kU[3] - kU[2])*kU[1]/2*(1 - ((kU[3] - z)/(kU[3] - kU[2]))^2)
    else
        return kU[1]*kU[2] - kU[1]^2/2 + (kU[3] - kU[2])*kU[1]/2
    end
end

function ψ(z::T1, spec::Hampel)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU

    if z < -kL[3]
        return 0.0
    elseif z < -kL[2]
        return -kL[1]*(kL[3] + z)/(kL[3] - kL[2])
    elseif z < -kL[1]
        return -kL[1]
    elseif z < kU[1]
        return z
    elseif z <= kU[2]
        return kU[1]
    elseif z <= kU[3]
        return kU[1]*(kU[3] - z)/(kU[3] - kU[2])
    else
        return 0.0
    end
end

function ψder(z::T1, spec::Hampel)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    if z < -kL[3]
        return 0.0
    elseif z < -kL[2]
        return -kL[1]/(kL[3] - kL[2])
    elseif z < -kL[1]
        return 0.0
    elseif z <= kU[1]
        return 1.0
    elseif z <= kU[2]
        return 0.0
    elseif z <= kU[3]
        return -kU[1]/(kU[3] - kU[2])
    else
        return 0.0
    end
end

function w(z::T1, spec::Hampel)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    if z < -kL[3]
        return 0.0
    elseif z < -kL[2]
        return -kL[1]*(kL[3] + z)/(kL[3] - kL[2])/z
    elseif z < -kL[1]
        return -kL[1]/z
    elseif z <= kU[1]
        return 1.0
    elseif z <= kU[2]
        return kU[1]/z
    elseif z <= kU[3]
        return kU[1]*(kU[3] - z)/(kU[3] - kU[2])/z
    else
        return 0.0
    end
end

#########################
### Smoothed Versions ###
#########################


### Preparation

# Smooth piecewise function by log-exp smoothing
# Kenjun Xia (2019) "smoothing globally ....", journal of comp. electronics

# Upper
function U(z::T1, zi::T2, ϵ::T3)::Float64 where {T1, T2, T3 <: Real}
    zi-ϵ*log(1 + exp((zi-z)/ϵ))
end

function Uder(z::T1, zi::T2, ϵ::T3)::Float64 where {T1, T2, T3 <: Real}
    exp(zi/ϵ)/(exp(z/ϵ) + exp(zi/ϵ))
end

function Us(z::T1, zi::T2, ϵ::T3)::Float64 where {T1, T2, T3 <: Real}
    ϵ*log((1 + exp((z + zi)/ϵ))/(1 + exp((z - zi)/ϵ))) - zi
end

function Usder(z::T1, zi::T2, ϵ::T3)::Float64 where {T1, T2, T3 <: Real}
    e1 = exp(z/ϵ)
    e2 = exp(zi/ϵ)
    e3 = exp((z + zi)/ϵ)
    e4 = exp(2*zi/ϵ)
    e1*(e4 - 1)/((e1 + e2)*(e3 + 1))
end

# Lower
function L(z::T1, zi::T2, ϵ::T3)::Float64 where {T1, T2, T3 <: Real}
    zi + ϵ*log(1 + exp((z-zi)/ϵ))
end

function Lder(z::T1, zi::T2, ϵ::T3)::Float64 where {T1, T2, T3 <: Real}
    exp(z/ϵ)/(exp(z/ϵ) + exp(zi/ϵ))
end

# Between
function B(z::T1, zL::T2, zR::T3, ϵ::T4)::Float64 where {T1, T2, T3, T4 <: Real}
    Us(U(z - zR, 0, ϵ), zR - zL, ϵ) + zR
end

function Bder(z::T1, zL::T2, zR::T3, ϵ::T4)::Float64 where {T1, T2, T3, T4 <: Real}
    Usder(U(z-zR, 0, ϵ), zR - zL, ϵ)*Uder(z - zR, 0, ϵ)
end


### Huber

function ψ(z::T1, spec::HuberS)::Float64 where {T1 <: Real}
    B(z, -spec.kL, spec.kU, spec.ϵ)
end

function ψder(z::T1, spec::HuberS)::Float64 where {T1 <: Real}
    Bder(z, -spec.kL, spec.kU, spec.ϵ)
end

function w(z::T1, spec::HuberS)::Float64 where {T1 <: Real}
    ψ(z, spec)/z
end

### Andrew

function ψ(z::T1, spec::AndrewS)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    ϵ = spec.ϵ
    out = kL/π*sin(π/kL*B(z, -kL, 0, ϵ))
    out += kU/π*sin(π/kU*B(z, 0, kU, ϵ))
    out
end

function ψder(z::T1, spec::AndrewS)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    ϵ = spec.ϵ
    out = cos(π/kL*B(z, -kL, 0, ϵ))*Bder(z, -kL, 0, ϵ)
    out += cos(π/kU*B(z, 0, kU, ϵ))*Bder(z, 0, kU, ϵ)
    out
end

function w(z::T1, spec::AndrewS)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    ϵ = spec.ϵ
    if abs(z) < ϵ
        return ψder(z, spec)
    end
    ψ(z, spec)/z
end

### Hampel

function ψ(z::T1, spec::HampelS)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    ϵ = spec.ϵ

    out = kL[1]*(kL[3] + B(z, -kL[3], -kL[2], ϵ))/(kL[3] - kL[2])*(-1)
    out += -kL[1]
    out += B(z, -kL[1], kU[1], ϵ)
    out += kU[1]
    out += kU[1]*(kU[3] - B(z, kU[2], kU[3], ϵ))/(kU[3] - kU[2])

    out -= 2*kL[1]*(-1)
    out -= 2*kU[1]
    out
end

function ψder(z::T1, spec::HampelS)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    ϵ = spec.ϵ

    out = (-1)*kL[1]/(kL[3] - kL[2])*Bder(z, -kL[3], -kL[2], ϵ)
    out += Bder(z, -kL[1], kU[1], ϵ)
    out += kU[1]/(kU[3] - kU[2])*Bder(z, kU[2], kU[3], ϵ)*(-1)
    out
end

function w(z::T1, spec::HampelS)::Float64 where {T1 <: Real}
    ψ(z, spec)/z
end
