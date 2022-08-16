# Compute the efficiency for given upper tuning constant (lower tc for unbiasedness)
# For multiple parameters: Single efficiencies and ignore covariances

function RAE(d::Distribution,
             spec::Vector{T};
             single::Bool = true) where {T <: MSetting}
        nPar = nParEff(d)
        Σm = AVar(d, spec)
        Σml = inv(FInfo(d))
        if single
                return diag(Σml) ./ diag(Σm)
        else
                return (det(Σml)/det(Σm))^(1/nPar)
        end
end

function RAE(d::Distribution,
             spec::T;
             single::Bool = true) where {T <: MSetting}
        nPar = nParEff(d)
        if nPar > 1
                return RAE(d, fill(spec, nPar), single = single)
        end
        Σm = AVar(d, spec)[1, 1]
        Σml = inv(FInfo(d))
        Σml / Σm
end
