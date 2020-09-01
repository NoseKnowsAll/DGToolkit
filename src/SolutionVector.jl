"""
    struct SolutionVector{T} <: AbstractVector{T}
Solution vector stored as 3D vector evaluated at nodes/quadrature points.
`A[:,iS,iK]` is the solution evaluated for state iS and element iK
"""
struct SolutionVector{T} <: AbstractVector{T}
    data::Array{T,3}

    function SolutionVector(data::Array{T,3}) where {T}
        new{T}(data)
    end

    function SolutionVector{T}(s1,s2,s3) where {T}
        new{T}(Array{T,3}(undef, s1,s2,s3))
    end
end

Base.similar(u::SolutionVector, ::Type{T}) where {T} = SolutionVector(Base.similar(u.data, T))

"""
    size(A::SolutionVector)
Return a tuple containing the dimensions of underlying operator A
"""
function Base.size(A::SolutionVector)
    (n,ns,ne) = size(A.data)
    return (n*ns*ne,)
end

"""
    true_size(A::SolutionVector)
Return a tuple containing the dimensions of underlying data for A
"""
function true_size(A::SolutionVector)
    return size(A.data)
end
"""
    true_size(A::SolutionVector, i::Integer)
Return the size of underlying data for A in dimension i
"""
function true_size(A::SolutionVector, i::Integer)
    return size(A.data,i)
end

"""
    getindex(A::SolutionVector, i::Integer)
Linear scalar indexing into array: access A[i]
"""
@inline function Base.getindex(A::SolutionVector, i::Integer)
    @boundscheck checkbounds(A.data, i)
    return A.data[i]
end

"""
    setindex!(A::SolutionVector, v, i::Integer)
Linear scalar indexed assignment: set A[i] = v
"""
@inline function Base.setindex!(A::SolutionVector, v, i::Integer)
    @boundscheck checkbounds(A.data, i)
    setindex!(A.data, v, i)
end

(==)(A::SolutionVector, B::SolutionVector) = A.data == B.data
(-)(A::SolutionVector) = SolutionVector(-A.data)
(+)(A::SolutionVector, B::SolutionVector) = SolutionVector(A.data+B.data)
(-)(A::SolutionVector, B::SolutionVector) = SolutionVector(A.data-B.data)
(*)(c::Number, A::SolutionVector) = SolutionVector(c*A.data)
(*)(A::SolutionVector, c::Number) = SolutionVector(c*A.data)
(/)(A::SolutionVector, c::Number) = SolutionVector(A.data / c)

"""
    lp_error!(u::SolutionVector, u_true::SolutionVector, p::Real=2)
Compute the lp-norm error ||u-u_true||_p of a SolutionVector (see: lp_norm).
Reuses the storage in u_true to compute this error.
"""
function lp_error!(u::SolutionVector, u_true::SolutionVector, p::Real=2)
    u_true -= u
    return lp_norm(u_true, p)
end

"""
    lp_norm(u::SolutionVector, p::Real=2)
Compute the lp-norm ||u||_p of a SolutionVector (see: norm).
If u has multiple states, then return array of L2 norms for each state.
"""
function lp_norm(u::SolutionVector, p::Real=2)
    (n,ns,ne) = true_size(u)
    lp_norms = zeros(ns)
    for iS = 1:ns
        @views lp_norms[iS] += LinearAlgebra.norm(vec(u.data[:,iS,:]),p)
    end
    return lp_norms
end
