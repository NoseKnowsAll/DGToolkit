" Helper function: One-indexed version of rem(x,y) "
function rem1(x,y)
    return mod(x-1,y)+1
end

"""
    struct LocalMatrix{T} <: AbstractMatrix{T}
Block diagonal structure for local matrix. `A[:,:,s,iK]` is a block matrix for
state s and element iK
"""
struct LocalMatrix{T} <: AbstractMatrix{T}
    data::Array{T,4}

    function LocalMatrix(data::Array{T,4}) where {T}
        new{T}(data)
    end
end

"""
    size(A::LocalMatrix)
Returns a tuple containing the dimensions of actual operator A
"""
function Base.size(A::LocalMatrix)
    (m,n,ns,ne) = size(A.data)
    return (m*ns*ne, n*ns*ne)
end

"""
    getindex(A::LocalMatrix{T}, i::Integer)
Linear scalar indexing into array
"""
@inline function Base.getindex(A::LocalMatrix, i::Integer)
    @boundscheck checkbounds(A.data, i)
    return A.data[i]
end

"""
    getindex(A::LocalMatrix{T}, i::Integer, j::Integer)
2-dimensional scalar indexing into array
"""
@inline function Base.getindex(A::LocalMatrix{T}, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(Bool, A,i,j)
    (m,n,ns,ne) = size(A.data)
    im = rem1(i ,m ); i2 = div(i ,m )
    is = rem1(i2,ns); i3 = div(i2,ns)
    ie = rem1(i3,ne)
    jn = rem1(j ,n ); j2 = div(j ,n )
    js = rem1(j2,ns); j3 = div(j2,ns)
    je = rem1(j3,ne)
    if is==js && ie==je && checkbounds(Bool,A.data, im,jn,is,ie)
        return @inbounds A.data[im,jn,is,ie]
    else
        return zero(T)
    end
end

"""
    setindex!(A::LocalMatrix{T}, v, i::Integer)
Linear scalar indexed assignment
"""
@inline function Base.setindex!(A::LocalMatrix, v, i::Integer)
    @boundscheck checkbounds(A.data, i)
    setindex!(A.data, v, i)
end

"""
    setindex!(A::LocalMatrix{T}, v, i::Integer, j::Integer)
2-dimensional scalar indexed assignment
"""
@inline function Base.setindex!(A::LocalMatrix, v, i::Integer, j::Integer)
    @boundscheck checkbounds(A,i,j)
    (m,n,ns,ne) = size(A.data)
    im = rem1(i ,m ); i2 = div(i ,m )
    is = rem1(i2,ns); i3 = div(i2,ns)
    ie = rem1(i3,ne)
    jn = rem1(j ,n ); j2 = div(j ,n )
    js = rem1(j2,ns); j3 = div(j2,ns)
    je = rem1(j3,ne)
    if is==js && ie==je && checkbounds(Bool,A.data, im,jn,is,ie)
        @inbounds setindex!(A.data, v, im,jn,is,ie)
    elseif !iszero(v)
        throw(ArgumentError("Cannot set off-diagonal entry ($i,$j) to non-zero value $v"))
    end
    return v
end

"""
    issymmetric(A::LocalMatrix)
Tests whether a LocalMatrix is symmetric
"""
function LinearAlgebra.issymmetric(A::LocalMatrix)
    println("my issymmetric")
    all(LinearAlgebra.issymmetric, [@view A.data[:,:,i,j] for i=1:size(A.data,3), j=1:size(A.data,4)])
end

"""
    isdiag(A::LocalMatrix)
Tests whether a LocalMatrix is diagonal
"""
function LinearAlgebra.isdiag(A::LocalMatrix)
    println("my isdiag")
    all(LinearAlgebra.isdiag, [@view A.data[:,:,i,j] for i=1:size(A.data,3), j=1:size(A.data,4)])
end

(==)(A::LocalMatrix, B::LocalMatrix) = A.data == B.data
(-)(A::LocalMatrix) = LocalMatrix(-A.data)
(+)(A::LocalMatrix, B::LocalMatrix) = LocalMatrix(A.data+B.data)
(-)(A::LocalMatrix, B::LocalMatrix) = LocalMatrix(A.data-B.data)
(*)(c::Number, A::LocalMatrix) = LocalMatrix(c*A.data)
(*)(A::LocalMatrix, c::Number) = LocalMatrix(c*A.data)
(/)(A::LocalMatrix, c::Number) = LocalMatrix(A.data / c)
function (*)(A::LocalMatrix, B::LocalMatrix)
    println("my *")
    @assert size(A.data,1) == size(B.data,1) && size(A.data,2) == size(B.data,2) && size(A.data,4) == size(B.data,4)
    if size(A.data,3) == size(A.data,3)
        return @inbounds LocalMatrix([A.data[:,:,i,j]*B.data[:,:,i,j] for i=1:size(A.data,3), j=1:size(A.data,4)])
    elseif size(A.data,3) == 1
        return @inbounds LocalMatrix([A.data[:,:,1,j]*B.data[:,:,i,j] for i=1:size(B.data,3), j=1:size(A.data,4)])
    elseif size(B.data,3) == 1
        return @inbounds LocalMatrix([A.data[:,:,i,j]*B.data[:,:,1,j] for i=1:size(A.data,3), j=1:size(A.data,4)])
    else
        throw(DimensionMismatch("Local matrices A ($(size(A.data))) and B ($(size(B.data)))"))
    end
end

function (*)(A::LocalMatrix{T1}, x::AbstractVector{T2}) where {T1, T2}
    println("my *1")
    (m,n,ns,ne) = size(A.data)
    b = Vector{typeof(zero(T1)*zero(T2))}(undef, m*ns*ne)
    for j = 1:ne
        for i = 1:ns
            offset = ns*(i-1)*(j-1)
            b[offset+m*(i-1)+1:offset+m*i] = A.data[:,:,i,j]*x[offset+n*(i-1)+1:offset+n*i]
        end
    end
    return b
end

function (*)(A::LocalMatrix{T1}, x::SolutionVector{T2}) where {T1, T2}
    println("my *2")
    (m,n,ns,ne) = size(A.data)
    (nx,nsx,nex) = size(x.data)
    @assert n == nx && ne == nex
    b = Array{typeof(zero(T1)*zero(T2)),3}(undef, m,nsx,ne)
    if ns == nsx
        for j = 1:ne
            for i = 1:nsx
                @inbounds b[:,i,j] = A.data[:,:,i,j]*x[:,i,j]
            end
        end
    elseif ns == 1
        for j = 1:ne
            for i = 1:nsx
                @inbounds b[:,i,j] = A.data[:,:,1,j]*x[:,i,j]
            end
        end
    else
        throw(DimensionMismatch("Local matrix A ($(size(A.data))) cannot multiply x ($(size(x.data)))"))
    end
    return SolutionVector(b)
end
