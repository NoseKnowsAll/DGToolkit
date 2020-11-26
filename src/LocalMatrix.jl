" Helper function: One-indexed version of rem(x,y) "
function rem1(x,y)
    return mod(x-1,y)+1
end

"""
    struct LocalMatrix{T} <: AbstractMatrix{T}
Block diagonal structure for local matrix. `A[:,:,iS,iK]` is a block matrix for
state iS and element iK
"""
struct LocalMatrix{T} <: AbstractMatrix{T}
    data::Array{T,4}
    factorizations::Array{Any,2}

    function LocalMatrix(data::Array{T,4}) where {T}
        new{T}(data,Array{Any}(undef, size(data,3), size(data,4)))
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
    getindex(A::LocalMatrix, i::Integer)
Linear scalar indexing into LocalMatrix
"""
@inline function Base.getindex(A::LocalMatrix, i::Integer)
    (m,n,ns,ne) = size(A.data)
    j = rem1(i, m*ns*ne); i = div(i, m*ns*ne, RoundUp)
    return getindex(A,i,j)
end

"""
    getindex(A::LocalMatrix{T}, i::Integer, j::Integer)
2-dimensional scalar indexing into LocalMatrix
"""
@inline function Base.getindex(A::LocalMatrix{T}, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(Bool, A,i,j)
    (m,n,ns,ne) = size(A.data)
    im = rem1(i ,m ); i2 = div(i ,m , RoundUp)
    is = rem1(i2,ns); i3 = div(i2,ns, RoundUp)
    ie = rem1(i3,ne)
    jn = rem1(j ,n ); j2 = div(j ,n , RoundUp)
    js = rem1(j2,ns); j3 = div(j2,ns, RoundUp)
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
    im = rem1(i ,m ); i2 = div(i ,m , RoundUp)
    is = rem1(i2,ns); i3 = div(i2,ns, RoundUp)
    ie = rem1(i3,ne)
    jn = rem1(j ,n ); j2 = div(j ,n , RoundUp)
    js = rem1(j2,ns); j3 = div(j2,ns, RoundUp)
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
    all(LinearAlgebra.issymmetric, [@view A.data[:,:,iS,iK] for iS=1:size(A.data,3), iK=1:size(A.data,4)])
end

"""
    isdiag(A::LocalMatrix)
Tests whether a LocalMatrix is diagonal
"""
function LinearAlgebra.isdiag(A::LocalMatrix)
    all(LinearAlgebra.isdiag, [@view A.data[:,:,iS,iK] for iS=1:size(A.data,3), iK=1:size(A.data,4)])
end

(==)(A::LocalMatrix, B::LocalMatrix) = A.data == B.data
(-)(A::LocalMatrix) = LocalMatrix(-A.data)
(+)(A::LocalMatrix, B::LocalMatrix) = LocalMatrix(A.data+B.data)
(-)(A::LocalMatrix, B::LocalMatrix) = LocalMatrix(A.data-B.data)
(*)(c::Number, A::LocalMatrix) = LocalMatrix(c*A.data)
(*)(A::LocalMatrix, c::Number) = LocalMatrix(c*A.data)
(/)(A::LocalMatrix, c::Number) = LocalMatrix(A.data / c)
function (*)(A::LocalMatrix{T1}, B::LocalMatrix{T2}) where {T1,T2}
    (ma,na,nsa,nea) = size(A.data)
    (mb,nb,nsb,neb) = size(B.data)
    @assert ma == mb && na == nb && nea == neb
    data = Array{typeof(zero(T1)*zero(T2)),4}(undef, ma,na,max(nsa,nsb),nea)
    if nsa == nsb
        for iK = 1:nea
            for iS = 1:nsa
                @inbounds @views data[:,:,iS,iK] = A.data[:,:,iS,iK]*B.data[:,:,iS,iK]
            end
        end
    elseif nsa == 1
        for iK = 1:nea
            for iS = 1:nsb
                @inbounds @views data[:,:,iS,iK] = A.data[:,:,1,iK]*B.data[:,:,iS,iK]
            end
        end
    elseif nsb == 1
        for iK = 1:nea
            for iS = 1:nsa
                @inbounds @views data[:,:,iS,iK] = A.data[:,:,iS,iK]*B.data[:,:,1,iK]
            end
        end
    else
        throw(DimensionMismatch("Local matrices A $(size(A.data)) and B $(size(B.data))"))
    end
    return LocalMatrix(data)
end

"""
    mul!(C::LocalMatrix, A::LocalMatrix, B::LocalMatrix, α::Number, β::Number)
Return A B α + C β by overwriting C
"""
function LinearAlgebra.mul!(C::LocalMatrix{T3}, A::LocalMatrix{T1}, B::LocalMatrix{T2}, α::Number, β::Number) where {T1,T2,T3}
    @assert T3 == typeof(zero(T1)*zero(T2))
    (ma,na,nsa,nea) = size(A.data)
    (mb,nb,nsb,neb) = size(B.data)
    (mc,nc,nsc,nec) = size(C.data)
    @assert ma == mb == mc && na == nb == nc && nea == neb == nec
    if nsa == nsb == nsc
        C.data .*= β
        for iK = 1:nea
            for iS = 1:nsa
                @inbounds @views C.data[:,:,iS,iK] += α*A.data[:,:,iS,iK]*B.data[:,:,iS,iK]
            end
        end
    elseif nsa == 1 && nsb == nsc
        C.data .*= β
        for iK = 1:nea
            for iS = 1:nsb
                @inbounds @views C.data[:,:,iS,iK] += α*A.data[:,:,1,iK]*B.data[:,:,iS,iK]
            end
        end
    elseif nsb == 1 && nsa == nsc
        C.data .*= β
        for iK = 1:nea
            for iS = 1:nsa
                @inbounds @views C.data[:,:,iS,iK] += α*A.data[:,:,iS,iK]*B.data[:,:,1,iK]
            end
        end
    else
        throw(DimensionMismatch("Local matrices A $(size(A.data)), B $(size(B.data)), and C $(size(C.data))"))
    end
    return C
end

function (*)(A::LocalMatrix{T1}, x::AbstractVector{T2}) where {T1, T2}
    (m,n,ns,ne) = size(A.data)
    b = Vector{typeof(zero(T1)*zero(T2))}(undef, m*ns*ne)
    for iK = 1:ne
        for iS = 1:ns
            offset = ns*(iS-1)*(iK-1)
            @views b[offset+m*(iS-1)+1:offset+m*iS] = A.data[:,:,iS,iK]*x[offset+n*(iS-1)+1:offset+n*iS]
        end
    end
    return b
end

"""
    mul!(y::SolutionVector, A::LocalMatrix, x::SolutionVector, α::Number, β::Number)
Return A x α + y β by overwriting y
"""
function LinearAlgebra.mul!(y::SolutionVector, A::LocalMatrix, x::SolutionVector, α::Number, β::Number)
    (ma,na,nsa,nea) = size(A.data)
    (nx,nsx,nex) = size(x.data)
    (ny,nsy,ney) = size(y.data)
    @assert ma == ny && na == nx && nea == nex == ney
    if nsa == nsx == nsy
        y.data .*= β
        for iK = 1:nea
            for iS = 1:nsa
                @inbounds @views y.data[:,iS,iK] += α*A.data[:,:,iS,iK]*x.data[:,iS,iK]
            end
        end
    elseif nsa == 1 && nsx == nsy
        y.data .*= β
        for iK = 1:nea
            for iS = 1:nsx
                @inbounds @views y.data[:,iS,iK] += α*A.data[:,:,1,iK]*x.data[:,iS,iK]
            end
        end
    else
        throw(DimensionMismatch("Local matrix A $(size(A.data)), SolutionVector x $(size(x.data)) and y $(size(y.data))"))
    end
    return y
end

function (*)(A::LocalMatrix{T1}, x::SolutionVector{T2}) where {T1, T2}
    (m,n,ns,ne) = size(A.data)
    (nx,nsx,nex) = size(x.data)
    @assert n == nx && ne == nex
    b = Array{typeof(zero(T1)*zero(T2)),3}(undef, m,nsx,ne)
    if ns == nsx
        for iK = 1:ne
            for iS = 1:nsx
                @inbounds @views b[:,iS,iK] = A.data[:,:,iS,iK]*x.data[:,iS,iK]
            end
        end
    elseif ns == 1
        for iK = 1:ne
            for iS = 1:nsx
                @inbounds @views b[:,iS,iK] = A.data[:,:,1,iK]*x.data[:,iS,iK]
            end
        end
    else
        throw(DimensionMismatch("Local matrix A $(size(A.data)) cannot multiply x $(size(x.data))"))
    end
    return SolutionVector(b)
end

"""
    factorize(A::LocalMatrix)
Explicitly compute the block-diagonal full LU factorizations of A
"""
function factorize!(A::LocalMatrix)
    (m,n,ns,ne) = size(A.data)
    @assert m == n "Block diagonals of LocalMatrix must be square!"
    for iK = 1:ne
        for iS = 1:ns
            A.factorizations[iS,iK] = LinearAlgebra.lu(A.data[:,:,iS,iK])
        end
    end
end

"""
    isfactorized(A::LocalMatrix)
Return whether this LocalMatrix has been factorized yet
"""
isfactorized(A::LocalMatrix) = isassigned(A.factorizations,1)

"""
    ldiv!(A::LocalMatrix, x::SolutionVector)
In-place linear solve A\\x using block-diagonal LU factorizations. Compute this
block-diagonal factorization if not yet computed.
"""
function LinearAlgebra.ldiv!(A::LocalMatrix, x::SolutionVector)
    println("my ldiv!")
    (m,n,ns,ne) = size(A.data)
    (nx,nsx,nex) = size(x.data)
    @assert n == nx && ne == nex && m == n
    if !isfactorized(A)
        factorize!(A)
    end
    if ns == nsx
        for iK = 1:ne
            for iS = 1:nsx
                @views LinearAlgebra.ldiv!(A.factorizations[iS,iK], x.data[:,iS,iK])
            end
        end
    elseif ns == 1
        for iK = 1:ne
            for iS = 1:nsx
                @views LinearAlgebra.ldiv!(A.factorizations[1,iK], x.data[:,iS,iK])
            end
        end
    else
        throw(DimensionMismatch("Local matrix A $(size(A.data)) cannot \\ x $(size(x.data))"))
    end
    x
end

"""
    A::LocalMatrix \\ x::SolutionVector
Linear solve A\\x using block-diagonal LU factorizations. Compute this
block-diagonal factorization if not yet computed.
"""
function \(A::LocalMatrix, x::SolutionVector)
    println("my \\")
    (m,n,ns,ne) = size(A.data)
    (nx,nsx,nex) = size(x.data)
    @assert n == nx && ne == nex && m == n
    b = deepcopy(x)
    LinearAlgebra.ldiv!(A, b)
end

"""
    L2_error(u::SolutionVector, u_true::SolutionVector, M::LocalMatrix)
Compute the L2 error ||u-u_true||_{L_2} = (u-u_true)'*M*(u-u_true) of a
SolutionVector (see: L2_norm). Reuses the storage in u_true to compute this error.
"""
function L2_error!(u::SolutionVector, u_true::SolutionVector, M::LocalMatrix)
    u_true -= u
    return L2_norm(u_true, M)
end

"""
    L2_norm(u::SolutionVector, M::LocalMatrix)
Compute the L2 norm ||u||_{L_2} = u'*M*u of a SolutionVector. If u has multiple
states, then return array of L2 norms for each state.
"""
function L2_norm(u::SolutionVector, M::LocalMatrix)
    Mu = M*u
    (n,ns,ne) = true_size(Mu)
    l2_norms = zeros(ns)
    for iK = 1:ne
        for iS = 1:ns
            @views l2_norms[iS] += u.data[:,iS,iK]'*Mu.data[:,iS,iK]
        end
    end
    return l2_norms
end
