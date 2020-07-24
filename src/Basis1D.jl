"""
    Module Basis1D
General math tools for 1D bases used by DG
"""

module Basis1D

export gauss_lobatto_quad, gauss_quad
export jacobiP, grad_jacobiP
export vandermonde1D, grad_vandermonde1D
export chebyshev

using LinearAlgebra
using SpecialFunctions

"""
    gauss_lobatto_quad(p, α=0, β=0)
Initialize nodes and weights (x,w) of Gauss-Legendre-Lobatto (GLL) quadrature points.
"""
function gauss_lobatto_quad(p, α=0, β=0)

    if (α!=0) && (β!=0)
        error("currently only allows alpha == beta == 0")
    end
    x = zeros(p+1)
    w = zeros(p+1)
    if p == 0
        x[1] = 0
        w[1] = 2
    elseif p == 1
        x[1] = -1.0
        x[2] = 1.0
        w[1] = 1.0
        w[2] = 1.0
    else
        xint, w = gauss_quad(p-2, α+1, β+1)
        x = [-1 transpose(xint) 1]

        V = vandermonde_1D(p,x)
        w = vec(sum(inv(V*V'),dims=2))
    end
    return x[:],w[:]
end

"""
    gauss_quad(p, α, β)
Initialize nodes and weights (x,w) of Gaussian quadrature of Jacobi Polynomial (α,β)
"""
function gauss_quad(p, α=0, β=0)
    if p == 0
        x = [-(α-β)/(α+β+2)]
        w = [2]
        return x, w
    end

    J = zeros(p+1, p+1)
    h₁ = @. 2*(0:p)+α+β
    J = diagm(0 => (@. -1/2*(α^2-β^2)/(h₁+2)/h₁), 1 => (@. 2/(h₁[1:p]+2)*sqrt((1:p)*((1:p)+α+β)*((1:p)+α)*((1:p)+β)/(h₁[1:p]+1)/(h₁[1:p]+3))))
    if α+β<10*eps()
        J[1,1] = 0.0
    end
    J = J + transpose(J) # Finalize symmetric tridiagonal Jacobi matrix

    x, V = eigen(J)
    w = @. transpose(V[1,:])^2*2^(α+β+1)/(α+β+1)*gamma(α+1)*gamma(β+1)/gamma(α+β+1)
    return x[:], w[:]
end

"""
    grad_jacobiP(p, r, α, β)
Evaluates the derivative of Jacobi Polynomial (α, β) of order p at nodes r
"""
function grad_jacobiP(p, r, α, β)
    dP = zeros(length(r))
    if p != 0
        dP = sqrt(p*(p+α+β+1))*jacobiP(p-1,r,α+1,β+1)
    end
    return dP
end

"""
    jacobiP(p, x, α, β)
Evaluate Jacobi Polynomial (α, β) of order p at points x
"""
function jacobiP(p, x, α, β)
    xp = x
    if size(xp, 2) == 1
        xp = transpose(xp)
    end

    PL = zeros(p+1,length(xp))
    γ₀ = 2^(α+β+1)/(α+β+1)*gamma(α+1)*gamma(β+1)/gamma(α+β+1)
    PL[1,:] .= 1.0/sqrt(γ₀)
    if p == 0
        P = transpose(PL)
        return P
    end
    γ₁ = (α+1)*(β+1)/(α+β+3)*γ₀
    PL[2,:] = ((α+β+2).*xp/2 .+ (α-β)/2)/sqrt(γ₁)
    if p == 1
        P = PL[p+1,:]
        return P
    end

    aold = 2/(2+α+β)*sqrt((α+1)*(β+1)/(α+β+3))

    for i = 1:p-1
        h₁ = 2i+α+β
        anew = 2/(h₁+2)*sqrt((i+1)*(i+1+α+β)*(i+1+α)*(i+1+β)/(h₁+1)/(h₁+3))
        bnew = -(α^2-β^2)/h₁/(h₁+2)
        PL[i+2,:] = 1/anew*(-aold*transpose(PL[i,:]).+(xp.-bnew).*transpose(PL[i+1,:]))
        aold = anew
    end

    P = PL[p+1,:]
    return P;
end

"""
    vandermonde1D(p, r)
Initialize the 1D Vandermonde matrix of order p Legendre polynomials at nodes r
"""
function vandermonde1D(p, r)
    V1D = zeros(length(r), p+1)
    for j = 1:p+1
        V1D[:,j] = jacobiP(j-1, r[:], 0, 0)
    end
    return V1D
end

"""
    grad_vandermonde1D(p, r)
Initialize the 1D gradient Vandermonde matrix of order p Legendre polynomials at nodes r
"""
function grad_vandermonde1D(p, r)
    V1D = zeros(length(r), p+1)
    for j = 1:p+1
        V1D[:,j] = grad_jacobiP(j-1, r[:], 0, 0)
    end
    return V1D
end

"""
    chebyshev(p)
Returns the 1D Chebyshev nodes of order p on [-1,1]
"""
function chebyshev(p)
    cheby = cos.((p:-1:0)*π/p)
end

end
