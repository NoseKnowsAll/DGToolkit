"""
    Module Basis3D
Includes DG core functions for evaluating 3D bases on hexahedral elements
"""
module Basis3D

export vandermonde3D, grad_vandermonde3D
export chebyshev3D, equidistant_nodes3D, gauss_quad3D
export dPhi3D

import Basis1D
using LinearAlgebra, SparseArrays

"""
    vandermonde3D(p, x, y, z)
Return the 3D vandermonde matrix - the order 0:p Legendre polynomials
evaluated at 3D nodes (x,y,z)
"""
function vandermonde3D(p, x, y, z)
    V = Array{Float64,6}(undef,length(x),length(y),length(z),p+1,p+1,p+1)
    lx = Basis1D.vandermonde1D(p, x)
    ly = Basis1D.vandermonde1D(p, y)
    lz = Basis1D.vandermonde1D(p, z)
    for ipz = 1:p+1
        for ipy = 1:p+1
            for ipx = 1:p+1
                for iz = 1:length(z)
                    for iy = 1:length(y)
                        for ix = 1:length(x)
                            V[ix,iy,iz,ipx,ipy,ipz] = lx[ix,ipx]*ly[iy,ipy]*lz[iz,ipz]
                        end
                    end
                end
            end
        end
    end
    return V
end

"""
    grad_vandermonde3D(p, x, y, z)
Return the gradient of the order 0:p Legendre polynomials evaluated
at 3D nodes (x,y,z)
"""
function grad_vandermonde3D(p, x, y, z)
    dV = Array{Float64,7}(undef, length(x),length(y),length(z), p+1,p+1,p+1, 3)
    lx = Basis1D.vandermonde1D(p, x)
    ly = Basis1D.vandermonde1D(p, y)
    lz = Basis1D.vandermonde1D(p, z)
    dlx = Basis1D.grad_vandermonde1D(p, x)
    dly = Basis1D.grad_vandermonde1D(p, y)
    dlz = Basis1D.grad_vandermonde1D(p, z)
    # Derivative wrt x
    for ipz = 1:p+1
        for ipy = 1:p+1
            for ipx = 1:p+1
                for iz = 1:length(z)
                    for iy = 1:length(y)
                        for ix = 1:length(x)
                            dV[ix,iy,iz,ipx,ipy,ipz,1] = dlx[ix,ipx]*ly[iy,ipy]*lz[iz,ipz]
                        end
                    end
                end
            end
        end
    end
    # Derivative wrt y
    for ipz = 1:p+1
        for ipy = 1:p+1
            for ipx = 1:p+1
                for iz = 1:length(z)
                    for iy = 1:length(y)
                        for ix = 1:length(x)
                            dV[ix,iy,iz,ipx,ipy,ipz,2] = lx[ix,ipx]*dly[iy,ipy]*lz[iz,ipz]
                        end
                    end
                end
            end
        end
    end
    # Derivative wrt z
    for ipz = 1:p+1
        for ipy = 1:p+1
            for ipx = 1:p+1
                for iz = 1:length(z)
                    for iy = 1:length(y)
                        for ix = 1:length(x)
                            dV[ix,iy,iz,ipx,ipy,ipz,3] = lx[ix,ipx]*ly[iy,ipy]*dlz[iz,ipz]
                        end
                    end
                end
            end
        end
    end
    return dV
end

"""
    gauss_quad3D(p)
Return the 3D Gaussian quadrature points and weights (x,w) of order p on the domain [-1,1]^3
"""
function gauss_quad3D(p)
    (x1D,w1D) = Basis1D.gauss_quad(p)
    n = length(x1D)
    x3D = Array{Float64,4}(undef, 3, n,n,n)
    w3D = Array{Float64,3}(undef, n,n,n)
    for iz = 1:n
        for iy = 1:n
            for ix = 1:n
                x3D[1,ix,iy,iz] = x1D[ix]
                x3D[2,ix,iy,iz] = x1D[iy]
                x3D[3,ix,iy,iz] = x1D[iz]
                w3D[ix,iy,iz] = w1D[ix]*w1D[iy]*w1D[iz]
            end
        end
    end
    return (x3D, w3D)
end

"""
    chebyshev3D(p)
Return the 3D chebyshev points of order p on the domain [-1,1]^3
"""
function chebyshev3D(p)
    x1D = Basis1D.chebyshev(p)
    n = length(x1D)
    x3D = Array{Float64,4}(undef, 3, n,n,n)
    for iz = 1:n
        for iy = 1:n
            for ix = 1:n
                x3D[1,ix,iy,iz] = x1D[ix]
                x3D[2,ix,iy,iz] = x1D[iy]
                x3D[3,ix,iy,iz] = x1D[iz]
            end
        end
    end
    return x3D
end

"""
    equidistant_nodes3D(p)
Return the "order p" 3D points on the domain [-1,1]^3
"""
function equidistant_nodes3D(p)
    x1D = -1:2/p:1
    n = length(x1D)
    x3D = Array{Float64,4}(undef, 3, n,n,n)
    for iz = 1:n
        for iy = 1:n
            for ix = 1:n
                x3D[1,ix,iy,iz] = x1D[ix]
                x3D[2,ix,iy,iz] = x1D[iy]
                x3D[3,ix,iy,iz] = x1D[iz]
            end
        end
    end
    return x3D
end

"""
    interpolation_matrix3D(xyz_from, xyz_to)
Compute an interpolation matrix from a set of 3D points to another set of 3D points.
Assumes that points interpolating from provide enough accuracy (aka - they are
well spaced out and of high enough order), and define a cube. Points
interpolating onto can be of any size, but must be defined on this same cube.
Interpolation matrix ∈ ℜ^(3D size of xyz_to x 3D size of xyz_from)
"""
function interpolation_matrix3D(xyz_from, xyz_to)
    # Create nodal representation of reference bases
    (x_from, y_from, z_from) = xyz_from
    order = size(x_from,1) - 1 # Assumes order = (size of x_from) - 1
    @assert size(x_from,1) == size(y_from,1) == size(z_from,1)
    n_from = size(x_from,1)*size(y_from,1)*size(z_from,1)
    l_from = vandermonde3D(order, x_from, y_from, z_from)

    eye = diagm(0=>ones(n_from))
    V = reshape(l_from, n_from,n_from)
    coeffs_phi = V \ eye

    # Compute reference bases on the output points
    l_to = vandermonde3D(order, xyz_to...)
    n_to = prod(size.(xyz_to,1))
    V_to = reshape(l_to, n_to,n_from)

    # Construct interpolation matrix
    Interp3D = V_to*coeffs_phi
end

"""
    dPhi3D(xyz_from, xyz_to)
Compute the gradient of the basis functions defined by 3D points on another
set of 3D points.

Assumes that points interpolating from provide enough accuracy (aka - they are
well spaced out and of high enough order), and define a cube. Points
interpolating onto can be of any size, but must be defined on this same cube.
dPhi ∈ ℜ^(size of xyz_to × size of xyz_from × 3)
"""
function dPhi3D(xyz_from, xyz_to)
    dim = 3
    # Create nodal representation of reference bases
    (x_from, y_from, z_from) = xyz_from
    order = size(x_from,1) - 1 # Assumes order = (size of x_from) - 1
    @assert size(x_from,1) == size(y_from,1) == size(z_from,1)
    n_from = size(x_from,1)*size(y_from,1)*size(z_from,1)
    l_from = vandermonde3D(order, x_from, y_from, z_from)

    eye = diagm(0=>ones(n_from))
    V = reshape(l_from, n_from,n_from)
    coeffs_phi = V \ eye

    # Compute derivative of reference bases on the output points
    dl_to = grad_vandermonde3D(order, xyz_to...)
    n_to = prod(size.(xyz_to,1))
    dV = reshape(dl_to, n_to,n_from,dim)
    # Construct gradient of phi = dV*coeffs_phi
    dPhi_to = Array{Float64,3}(undef, n_to,n_from,dim)
    for l = 1:dim
        dPhi_to[:,:,l] = dV[:,:,l]*coeffs_phi
    end
    return dPhi_to
end

end
