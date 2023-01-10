# Function to check for convergence
# Should detect loops

function checkConvergence(mat::Matrix{T1},
                   i::Int64,
                   conv::T2 = 1e-05) where {T1 <: Real, T2 <: Real}
    diff = sum(abs.(mat[i, :] .- mat[i-1, :]))
    if i > 2
        diff = minimum([diff, sum(abs.(mat[i, :] .- mat[i-2, :]))])
    end
    if i > 3
        diff = minimum([diff, sum(abs.(mat[i, :] .- mat[i-3, :]))])
    end

    return diff < conv
end

function checkConvergence(vecc::Vector{T1},
                   i::Int64,
                   conv::T2 = 1e-05) where {T1 <: Real, T2 <: Real}
    diff = sum(abs(vecc[i] .- vecc[i-1]))
    if i > 2
        diff = minimum([diff, abs(vecc[i] .- vecc[i-2])])
    end
    if i > 3
        diff = minimum([diff, abs(vecc[i] .- vecc[i-3])])
    end
    
    return diff < conv
end
