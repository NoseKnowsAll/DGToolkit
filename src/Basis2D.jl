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
Return the 2D vandermonde matrix - the order 0:p Legendre polynomials
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
Return the gradient of the order 0:p Legendre polynomials
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
Return the 2D Gaussian quadrature points and weights (x,w) of order p on the domain [-1,1]^2
"""
function gauss_quad2D(p)
    (x1D,w1D) = Basis1D.gauss_quad(p)
    n = length(x1D)
    x2D = Array{Float64,3}(undef, 2, n,n)
    w2D = Array{Float64,2}(undef, n,n)
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
Return the 2D chebyshev points of order p on the domain [-1,1]^2
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
Return the "order p" 2D points on the domain [-1,1]^2
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

"""
    interpolation_matrix2D(xy_from, xy_to)
Compute an interpolation matrix from a set of 2D points to another set of 2D points.
Assumes that points interpolation from provide enough accuracy (aka - they are
well spaced out and of high enough order), and define a square. Points
interpolating onto can be of any size, but must be defined on this same square.
Interpolation matrix ∈ ℜ^(2D size of xy_to x 2D size of xy_from)
"""
function interpolation_matrix2D(xy_from, xy_to)
    # Create nodal representation of reference bases
    (x_from, y_from) = xy_from
    order = size(x_from,1) - 1 # Assumes order = (size of x_from) - 1
    @assert size(x_from,1) == size(y_from,1)
    n_from = size(x_from,1)*size(y_from,1)
    l_from = vandermonde2D(order, x_from, y_from)

    eye = diagm(0=>ones(n_from))
    V = reshape(l_from, n_from,n_from)
    coeffs_phi = V \ eye

    # Compute reference bases on the output points
    l_to = vandermonde2D(order, xy_to...)
    n_to = prod(size.(xy_to,1))
    V_to = reshape(l_to, n_to,n_from)

    # Construct interpolation matrix
    Interp2D = V_to*coeffs_phi
end

end
