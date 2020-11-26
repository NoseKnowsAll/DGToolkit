import Geometry

"""
    struct JacobianMatrix{T} <: AbstractMatrix{T}
Jacobian matrix. Block-diagonal local matrix represents intraelement operations,
off-diagonal represents connectivity between elements.
`diag[:,:,iS1,iS2,iK]` is a square matrix d(f(u[:,iS1,iK))/d(u[:,iS2,iK])
for states iS1, is2, element iK.
`offdiag[iN1,iFN2,iS1,iS2,iK1,iF]` is a rectangular matrix
d(f(u[iN1,iS1,iK1])/d(u[iFN2,iS2,mesh.e2e[iF,iK1]]), or effect of neighbor iF on
element iK1. For upper triangular portion of global matrix
(mesh.e2e[iF,iK1]<iK1), offdiag[:,:,iS1,iS2,iK1,iF] is actually the transpose of
values stored so that only one array need be used.
"""
struct JacobianMatrix{T} <: AbstractMatrix{T}
    diag::Array{T,5}
    offdiag::Array{T,6}
    mesh::Geometry.Mesh

    function JacobianMatrix(diag::Array{T,5}, offdiag::Array{T,6}, mesh::Geometry.Mesh) where {T}
        @assert size(diag,1) == size(diag,2) "Diagonal of Jacobian must be square!"
        @assert size(diag,3) == size(diag,4) "Number of Jacobian states must be consistent!"
        @assert size(offdiag,6) == Geometry.N_FACES "Number of Jacobian neighbors does not match geometry!"
        new{T}(diag,offdiag,mesh)
    end
end

"""
    size(A::JacobianMatrix)
Returns a tuple containing the dimensions of actual operator A
"""
function Base.size(A::JacobianMatrix)
    (m,n,ns,ns2,ne) = size(A.diag)
    return (m*ns*ne, n*ns*ne)
end
"""
    getindex(A::JacobianMatrix, i::Integer)
Linear scalar indexing into Jacobian matrix
"""
@inline function Base.getindex(A::JacobianMatrix, i::Integer)
    (m,n,ns,ns2,ne) = size(A.diag)
    j = rem1(i, m*ns*ne); i = div(i, m*ns*ne, RoundUp)
    return getindex(A,i,j)
end
"""
    getindex(A::JacobianMatrix{T}, i::Integer, j::Integer)
2-dimensional scalar indexing into JacobianMatrix
"""
@inline function Base.getindex(A::JacobianMatrix{T}, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(Bool, A,i,j)
    (m,n,ns,ne) = size(A.diag)
    im = rem1(i ,m ); i2 = div(i ,m , RoundUp)
    is = rem1(i2,ns); i3 = div(i2,ns, RoundUp)
    ie = rem1(i3,ne)
    jn = rem1(j ,n ); j2 = div(j ,n , RoundUp)
    js = rem1(j2,ns); j3 = div(j2,ns, RoundUp)
    je = rem1(j3,ne)
    if ie==je
        # On diagonal
        if checkbounds(Bool,A.diag, im,jn,is,js,ie)
            return @inbounds A.diag[im,jn,is,js,ie]
        end
    elseif je ∈ A.mesh.e2e[:,ie]
        iF = findfirst(x->x==je,A.mesh.e2e[:,ie])
        iFN = findfirst(x->x==im,A.mesh.ef2n[:,iF])
        iFN2 = 0
        if iFN != nothing
            # We need the neighbor's face node
            iFN2 = mesh.nfn2fn[iFN]
        else
            # Node is not on the face => no Jacobian dependence
            return zero(T)
        end
        if je < ie
            # Lower triangular part stored transposed
            return A.offdiag[iFN2,im,js,is,ie,iF]
        else
            # Upper triangular part stored directly
            return A.offdiag[im,iFN2,is,js,ie,iF]
        end
    else
        # Non-neighboring elements => no Jacbian dependence
        return zero(T)
    end
end
