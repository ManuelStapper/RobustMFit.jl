###############################################
### How to implement a custom loss function ###
###############################################

# For Example: A modification of Huber functions, where for |z| > k
# the loss function is not absolute, but logarithmic

# Create abstract type
abstract type MyTypeSetting <: MSetting end

# Create setting structs
mutable struct MyType <: MyTypeSetting
    kL::Float64
    kU::Float64
end

# Smooth version
mutable struct MyTypeS <: MyTypeSetting
    kL::Float64
    kU::Float64
    ϵ::Float64
end

# Create Constructors
function MyType(kL::T1, kU::T2; ϵ::T3 = 0.0) where {T1 <: Real, T2 <: Real, T3 <: Real}
    if kL <= 0
        error("Invalid lower tuning constant")
    end
    if kU <= 0
        error("Invalid upper tuning constant")
    end
    if ϵ < 0
        error("Invalid smoothing parameter ϵ")
    end
    if ϵ == 0.0
        return MyType(Float64(kL), Float64(kU))
    else
        return MyTypeS(Float64(kL), Float64(kU), Float64(ϵ))
    end
end

# Symmetric
function MyType(k::T1; ϵ::T2 = 0.0) where {T1, T2 <: Real}
    if k <= 0
        error("Invalid tuning constant")
    end
    if ϵ < 0
        error("Invalid smoothing parameter ϵ")
    end
    if ϵ == 0
        return MyType(Float64(k), Float64(k))
    else
        return MyTypeS(Float64(k), Float64(k), Float64(ϵ))
    end
end

# Implement functions
function ρ(z::T1, spec::T2)::Float64 where {T1 <: Real, T2 <: MyTypeSetting}
    kL = spec.kL
    kU = spec.kU
    # Include function here
    az = abs(z)
    if z < -kL
        return kL^2*(log(az) + 1/2 - log(kL))
    elseif z <= kU
        return az^2/2
    else
        return kU^2*(log(az) + 1/2 - log(kU))
    end
end

function ψ(z::T1, spec::MyType)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    # Include function here
    if z < -kL
        return kL^2/z
    elseif z <= kU
        return z
    else
        return kU^2/z
    end
end

function ψder(z::T1, spec::MyType)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    # Include function here
    if z < -kL
        return -kL^2/z^2
    elseif z <= kU
        return 1
    else
        return -kU^2/z^2
    end
end

function w(z::T1, spec::MyType)::Float64 where {T1 <: Real}
    ψ(z, spec)/z
end

# Smoothed functions
function ψ(z::T1, spec::MyTypeS)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    ϵ = spec.ϵ
    # Include function here
    out = kL^2/U(z, -kL, ϵ)
    out += B(z, -kL, kU, ϵ)
    out += kU^2/L(z, kU, ϵ)

    out += kL
    out -= kU
    out
end

function ψder(z::T1, spec::MyTypeS)::Float64 where {T1 <: Real}
    kL = spec.kL
    kU = spec.kU
    ϵ = spec.ϵ
    # Include function here
    out = -kL^2/U(z, -kL, ϵ)^2*Uder(z, -kL, ϵ)
    out += Bder(z, -kL, kU, ϵ)
    out -= kU^2/L(z, kU, ϵ)^2*Lder(z, kU, ϵ)

    out
end

function w(z::T1, spec::MyTypeS)::Float64 where {T1 <: Real}
    # Avoid numerical instability for z close to zero
    if abs(z) <= 0.01
        return ψder(z, spec)
    end
    ψ(z, spec)/z
end
