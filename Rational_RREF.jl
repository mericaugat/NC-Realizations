
using LinearAlgebra
using BenchmarkTools

export rref, rref!

#function rref!(A::Matrix{T}, ɛ=T <: Union{Rational,Integer} ? 0 : eps(norm(A,Inf))) where T
function rref!(A::Matrix{T}, ɛ=T <: Union{Rational,Integer} ? 0 : eps(norm(A,Inf))) where T
    nrows, ncols = size(A)
    i = j = 1
    while i <= nrows && j <= ncols
        (m, mi) = findmax(abs.(A[i:nrows,j]))
        mi = mi+i - 1
        if m <= ɛ
            if ɛ > 0
                A[i:nrows,j] .= zero(T)
            end
            j += 1
        else
            for k=j:ncols
                A[i, k], A[mi, k] = A[mi, k], A[i, k]
            end
            d = A[i,j]
            for k = j:ncols
                A[i,k] /= d
            end
            for k = 1:nrows
                if k != i
                    d = A[k,j]
                    for l = j:ncols
                        A[k,l] -= d*A[i,l]
                    end
                end
            end
            i += 1
            j += 1
        end
    end
    A
end

rrefconv(::Type{T}, A::Matrix) where {T} = rref!(copyto!(similar(A, T), A))




rref(A::Matrix{T}) where {T} = rref!(copy(A))
rref(A::Matrix{T}) where {T <: Complex} = rrefconv(ComplexF64, A)
rref(A::Matrix{ComplexF64}) = rref!(copy(A))
rref(A::Matrix{T}) where {T <: Union{Integer, Float16, Float32}} = rrefconv(Float64, A)
rref(A::AbstractMatrix) = rref(Matrix(A))


# The function rref(), but with the modification that it also returns the pivot vector

export rref_with_pivots, rref_with_pivots!

function rref_with_pivots!(A::Matrix{T}, ɛ=T <: Union{Rational,Integer} ? 0 : eps(norm(A,Inf))) where T
    nr, nc = size(A)
    pivots = Vector{Int64}()
    i = j = 1
    while i <= nr && j <= nc
        (m, mi) = findmax(abs.(A[i:nr,j]))
        mi = mi+i - 1
        if m <= ɛ
            if ɛ > 0
                A[i:nr,j] .= zero(T)
            end
            j += 1
        else
            for k=j:nc
                A[i, k], A[mi, k] = A[mi, k], A[i, k]
            end
            d = A[i,j]
            for k = j:nc
                A[i,k] /= d
            end
            for k = 1:nr
                if k != i
                    d = A[k,j]
                    for l = j:nc
                        A[k,l] -= d*A[i,l]
                    end
                end
            end
            append!(pivots,j)
            i += 1
            j += 1
        end
    end
    return A, pivots
end

rref_with_pivots_conv(::Type{T}, A::Matrix) where {T} = rref_with_pivots!(copyto!(similar(A, T), A))





rref_with_pivots(A::Matrix{T}) where {T} = rref_with_pivots!(copy(A))
rref_with_pivots(A::Matrix{T}) where {T <: Complex} = rref_with_pivots_conv(ComplexF64, A)
rref_with_pivots(A::Matrix{ComplexF64}) = rref_with_pivots!(copy(A))
rref_with_pivots(A::Matrix{T}) where {T <: Union{Integer, Float16, Float32}} = rref_with_pivots_conv(Float64, A)

rref_with_pivots(A::AbstractMatrix) = rref_with_pivots(Matrix(A))