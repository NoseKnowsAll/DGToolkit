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
end

"""
    size(A::SolutionVector)
Returns a tuple containing the dimensions of actual operator A
"""
function Base.size(A::SolutionVector)
    (n,ns,ne) = size(A.data)
    return (n*ns*ne,)
end

"""
    getindex(A::SolutionVector, i::Integer)
Linear scalar indexing into array
"""
@inline function Base.getindex(A::SolutionVector, i::Integer)
    @boundscheck checkbounds(A.data, i)
    return A.data[i]
end

"""
    setindex!(A::SolutionVector, v, i::Integer)
Linear scalar indexed assignment
"""
@inline function Base.setindex!(A::SolutionVector, v, i::Integer)
    @boundscheck checkbounds(A.data, i)
    setindex!(A.data, v, i)
end
