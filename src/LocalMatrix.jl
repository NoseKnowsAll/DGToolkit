" One-indexed version of div(x,y) TODO: unnecessary?"
function div1(x::Integer,y::Integer)
    return floor(Int,x/y)
end

" One-indexed version of rem(x,y) "
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

    function LocalMatrix{T}(data) where {T}
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
