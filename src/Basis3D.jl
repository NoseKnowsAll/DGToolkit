"""
    Module Basis3D
Includes DG core functions for evaluating 3D bases on hexahedral elements
"""
module Basis3D

export vandermonde3D, grad_vandermonde3D
export chebyshev3D, equidistant_nodes3D, gauss_quad3D

import Basis1D
using LinearAlgebra, SparseArrays

"""
    vandermonde3D(p, x, y, z)
Returns the 3D vandermonde matrix - the order 0:p Legendre polynomials
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
Returns the gradient of the order 0:p Legendre polynomials
evaluated at 3D nodes (x,y,z)
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
Returns the 3D Gaussian quadrature points and weights (x,w) of order p on the domain [-1,1]^3
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
Returns the 3D chebyshev points of order p on the domain [-1,1]^3
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
Returns the "order p" 3D points on the domain [-1,1]^3
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

end
