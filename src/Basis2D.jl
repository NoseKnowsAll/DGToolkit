"""
    Module Basis2D
Includes DG core functions for evaluating 2D bases on quads
"""
module Basis2D

export vandermonde2D, grad_vandermonde2D
export chebyshev2D, equidistant_nodes2D, gauss_quad2D

import Basis1D
using LinearAlgebra, SparseArrays

"""
    vandermonde2D(p, x, y)
Returns the 2D vandermonde matrix - the order 0:p Legendre polynomials
evaluated at 2D nodes (x,y)
"""
function vandermonde2D(p, x, y)
    V = Array{Float64,4}(undef,length(x),length(y),p+1,p+1)
    lx = Basis1D.vandermonde1D(p, x)
    ly = Basis1D.vandermonde1D(p, y)
    for ipy = 1:p+1
        for ipx = 1:p+1
            V[:,:,ipx,ipy] = lx[:,ipx]*(ly[:,ipy]')
        end
    end
    return V
end

"""
    grad_vandermonde2D(p, x, y)
Returns the gradient of the order 0:p Legendre polynomials
evaluated at 2D nodes (x,y)
"""
function grad_vandermonde2D(p, x, y)
    dV = Array{Float64,5}(undef, length(x),length(y), p+1,p+1, 2)
    lx = Basis1D.vandermonde1D(p, x)
    ly = Basis1D.vandermonde1D(p, y)
    dlx = Basis1D.grad_vandermonde1D(p, x)
    dly = Basis1D.grad_vandermonde1D(p, y)
    # Derivative wrt x
    for ipy = 1:p+1
        for ipx = 1:p+1
            dV[:,:,ipx,ipy,1] = dlx[:,ipx]*(ly[:,ipy]')
        end
    end
    # Derivative wrt y
    for ipy = 1:p+1
        for ipx = 1:p+1
            dV[:,:,ipx,ipy,2] = lx[:,ipx]*(dly[:,ipy]')
        end
    end
    return dV
end

"""
    gauss_quad2D(p)
Returns the 2D Gaussian quadrature points and weights (x,w) of order p on the domain [-1,1]^2
"""
function gauss_quad2D(p)
    (x1D,w1D) = Basis1D.gauss_quad(p)
    n = length(x1D)
    x2D = Array{Float64,3}(undef, 2, n, n)
    w2D = Array{Float64,2}(undef, n, n)
    for iy = 1:n
        for ix = 1:n
            x2D[1,ix,iy] = x1D[ix]
            x2D[2,ix,iy] = x1D[iy]
            w2D[ix,iy] = w1D[ix]*w1D[iy]
        end
    end
    return (x2D, w2D)
end

"""
    chebyshev2D(p)
Returns the 2D chebyshev points of order p on the domain [-1,1]^2
"""
function chebyshev2D(p)
    x1D = Basis1D.chebyshev(p)
    n = length(x1D)
    x2D = Array{Float64,3}(undef, 2, n, n)
    for iy = 1:n
        for ix = 1:n
            x2D[1,ix,iy] = x1D[ix]
            x2D[2,ix,iy] = x1D[iy]
        end
    end
    return x2D
end

"""
    equidistant_nodes2D(p)
Returns the "order p" 2D points on the domain [-1,1]^2
"""
function equidistant_nodes2D(p)
    x1D = -1:2/p:1
    n = length(x1D)
    x2D = Array{Float64,3}(undef, 2, n, n)
    for iy = 1:n
        for ix = 1:n
            x2D[1,ix,iy] = x1D[ix]
            x2D[2,ix,iy] = x1D[iy]
        end
    end
    return x2D
end

end
