import Geometry

"""
    struct JacobianMatrix{T} <: AbstractMatrix{T}
Jacobian matrix. Block-diagonal local matrix represents intraelement operations,
off-diagonal represents connectivity between elements.
`diag[:,iS1,:,iS2,iK]` is a square matrix d(f(u[:,iS1,iK))/d(u[:,iS2,iK])
for states iS1, iS2, element iK. Designed so that `diag[:,:,:,:,iK]` can be
in-place reshaped for matrix multiplication.
`offdiag[iN1,iS1,iFN2,iS2,iF,iK1]` is a rectangular matrix
d(f(u[iN1,iS1,iK1])/d(u[iFN2,iS2,mesh.e2e[iF,iK1]]), or effect of neighbor iF on
element iK1. Again, can in-place reshape for matrix multiplication. Note that
for upper triangular portion of global matrix (`mesh.e2e[iF,iK1]<iK1`),
the matrix `offdiag[:,:,:,:,iF,iK1]` actually yields the transpose of
operator needed; this format needs only one array for all off-diagonals.
"""
struct JacobianMatrix{T} <: AbstractMatrix{T}
    diag::Array{T,5}
    offdiag::Array{T,6}
    mesh::Geometry.Mesh

    function JacobianMatrix(diag::Array{T,5}, offdiag::Array{T,6}, mesh::Geometry.Mesh) where {T}
        @assert size(diag,1) == size(diag,3) "Diagonal of Jacobian must be square!"
        @assert size(diag,2) == size(diag,4) "Number of Jacobian states must be consistent!"
        @assert size(offdiag,5) == Geometry.N_FACES "Number of Jacobian neighbors does not match geometry!"
        new{T}(diag,offdiag,mesh)
    end
end

"""
    size(A::JacobianMatrix)
Returns a tuple containing the dimensions of actual operator A
"""
function Base.size(A::JacobianMatrix)
    (m,ns,n,ns2,ne) = size(A.diag)
    return (m*ns*ne, n*ns2*ne)
end
"""
    getindex(A::JacobianMatrix, i::Integer)
Linear scalar indexing into Jacobian matrix
"""
@inline function Base.getindex(A::JacobianMatrix, i::Integer)
    (sz1,sz2) = size(A)
    j = rem1(i, sz1); i = div(i, sz2, RoundUp)
    return getindex(A,i,j)
end
"""
    getindex(A::JacobianMatrix{T}, i::Integer, j::Integer)
2-dimensional scalar indexing into JacobianMatrix. Not efficient.
"""
@inline function Base.getindex(A::JacobianMatrix{T}, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(Bool, A,i,j)
    (m,ns,n,ns2,ne) = size(A.diag)
    im = rem1(i ,m ); i2 = div(i ,m , RoundUp)
    is = rem1(i2,ns); i3 = div(i2,ns, RoundUp)
    ie = rem1(i3,ne)
    jn = rem1(j ,n ); j2 = div(j ,n , RoundUp)
    js = rem1(j2,ns); j3 = div(j2,ns, RoundUp)
    je = rem1(j3,ne)
    if ie==je
        # On diagonal
        if checkbounds(Bool,A.diag, im,is,jn,js,ie)
            return @inbounds A.diag[im,is,jn,js,ie]
        end
    elseif je ∈ A.mesh.e2e[:,ie]
        iF = findfirst(x->x==je,A.mesh.e2e[:,ie])
        if je < ie
            # Lower triangular part stored transposed
            iFN2 = findfirst(x->x==jn,A.mesh.ef2n[:,iF])
            iFN = 0
            if !isnothing(iFN2)
                # We need the original face node
                iFN = mesh.nfn2fn[iFN2]
            else
                # Node is not on the face => no Jacobian dependence
                return zero(T)
            end
            return A.offdiag[jn,js,iFN,is,iF,ie]
        else
            # Upper triangular part stored directly
            iFN = findfirst(x->x==im,A.mesh.ef2n[:,iF])
            iFN2 = 0
            if !isnothing(iFN)
                # We need the neighbor's face node
                iFN2 = mesh.nfn2fn[iFN]
            else
                # Node is not on the face => no Jacobian dependence
                return zero(T)
            end
            return A.offdiag[im,is,iFN2,js,iF,ie]
        end
    else
        # Non-neighboring elements => no Jacobian dependence
        return zero(T)
    end
end
"""
    setindex!(A::JacobianMatrix, v, i::Integer)
Linear scalar indexed assignment of `JacobianMatrix`
"""
@inline function Base.setindex!(A::JacobianMatrix, v, i::Integer)
    (sz1,sz2) = size(A)
    j = rem1(i, sz1); i = div(i, sz2, RoundUp)
    setindex!(A, v, i, j)
end
"""
    setindex!(A::JacobianMatrix{T}, v, i::Integer, j::Integer)
2-dimensional scalar assignment of `JacobianMatrix`. Not efficient.
"""
@inline function Base.setindex!(A::JacobianMatrix{T}, v, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(Bool, A,i,j)
    (m,ns,n,ns2,ne) = size(A.diag)
    im = rem1(i ,m ); i2 = div(i ,m , RoundUp)
    is = rem1(i2,ns); i3 = div(i2,ns, RoundUp)
    ie = rem1(i3,ne)
    jn = rem1(j ,n ); j2 = div(j ,n , RoundUp)
    js = rem1(j2,ns); j3 = div(j2,ns, RoundUp)
    je = rem1(j3,ne)
    if ie==je
        # On diagonal
        if checkbounds(Bool,A.diag, im,is,jn,js,ie)
            @inbounds setindex!(A.diag, v, im,is,jn,js,ie)
        end
    elseif je ∈ A.mesh.e2e[:,ie]
        iF = findfirst(x->x==je,A.mesh.e2e[:,ie])
        if je < ie
            # Lower triangular part stored transposed
            iFN2 = findfirst(x->x==jn,A.mesh.ef2n[:,iF])
            iFN = 0
            if !isnothing(iFN2)
                # We need the original face node
                iFN = mesh.nfn2fn[iFN2]
            else
                if !iszero(v)
                    # Node is not on the face => no Jacobian dependence
                    throw(ArgumentError("Cannot set off-diagonal entry ($i,$j) to non-zero value $v"))
                else
                    return v
                end
            end
            return setindex!(A.offdiag, v, jn,js,iFN,is,iF,ie)
        else
            # Upper triangular part stored directly
            iFN = findfirst(x->x==im,A.mesh.ef2n[:,iF])
            iFN2 = 0
            if !isnothing(iFN)
                # We need the neighbor's face node
                iFN2 = mesh.nfn2fn[iFN]
            else
                if !iszero(v)
                    # Node is not on the face => no Jacobian dependence
                    throw(ArgumentError("Cannot set off-diagonal entry ($i,$j) to non-zero value $v"))
                else
                    return v
                end
            end
            return setindex!(A.offdiag, v, im,is,iFN2,js,iF,ie)
        end
    else
        # Non-neighboring elements => no Jacobian dependence
        if !iszero(v)
            throw(ArgumentError("Cannot set off-diagonal entry ($i,$j) to non-zero value $v"))
        else
            return v
        end
    end
end

(==)(A::JacobianMatrix, B::JacobianMatrix) = (A.diag == B.diag && A.offdiag == B.offdiag)
(-)(A::JacobianMatrix) = JacobianMatrix(-A.diag, -A.offdiag, A.mesh)
function (+)(A::JacobianMatrix, B::JacobianMatrix)
    @assert A.mesh === B.mesh
    JacobianMatrix(A.diag+B.diag, A.offdiag+B.offdiag, A.mesh)
end
function (-)(A::JacobianMatrix, B::JacobianMatrix)
    @assert A.mesh === B.mesh
    JacobianMatrix(A.diag-B.diag,A.offdiag-B.offdiag,mesh)
end
# TODO: Matrix addition with LocalMatrix as well?
(*)(c::Number, A::JacobianMatrix) = JacobianMatrix(c*A.diag, c*A.offdiag, A.mesh)
(*)(A::JacobianMatrix, c::Number) = JacobianMatrix(c*A.diag, c*A.offdiag, A.mesh)
(/)(A::JacobianMatrix, c::Number) = JacobianMatrix(A.diag / c, A.offdiag / c, A.mesh)
(*)(A::JacobianMatrix, B::JacobianMatrix) = error("Too expensive an operation!")

function (*)(A::JacobianMatrix{T1}, x::AbstractVector{T2}) where {T1, T2}
    (m,ns,n,ns2,ne) = size(A.diag)
    b = zeros(typeof(zero(T1)*zero(T2)), m*ns*ne)
    b_shaped = reshape(b, m*ns,ne)
    x_shaped = reshape(x, n*ns2,ne)
    for iK = 1:ne
        # Diagonal of A
        @views Adiag = reshape(A.diag[:,:,:,:,iK])
        @views b_shaped[:,iK] += Adiag*x_shaped[:,iK]
        # Off-diagonal of A
        # TODO: Everything below is completely unfinished
        for iF = 1:Geometry.N_FACES
            nK = A.mesh.e2e[iF,iK]
            if nK > 0
                if nK < iK
                    for iS2 = 1:ns2
                        for iS = 1:ns
                            @views b_shaped[:,iS,iK] += A.offdiag[:,:]TODO
                        end
                    end
                else

                end
            end
        end
    end
    return b
end
