export NavierStokes

"""
    struct NavierStokes{N} <: Application{N}
Defines the compressible Navier-Stokes equations:
∂ρ/∂t + ∇⋅(ρu) = 0
∂(ρu)/∂t + ∇⋅(ρu ⊗ u' + pI - τ) = 0
∂(ρE)/∂t + ∇⋅((ρE+p)u + q - τu) = 0
where ρ is the fluid density, u is the fluid velocity, E is the total energy,
p is the pressure, I is the N × N identity tensor, τ is the viscous stress
tensor
τ := μ (∇u + ∇u' - 2/3(∇⋅u)I),
with μ=1/Re the viscosity coefficient, and q is the heat flux
q := -μ/Pr ∇(E + p/ρ - 1/2||u||_2^2) = -γ/(Pr*Re) ∇(E+1/2||u||_2^2),
with Pr the Prandtl number (typically 0.72).
"""
struct NavierStokes{N} <: Application{N}
    # Constants TODO
    " Reynolds number "
    Re
    " Prandtl number "
    Pr
    NavierStokes{N}(Re,Pr) where {N} = new{N}(Re,Pr)
    function NavierStokes{N}(param_dict::Dict) where {N}
        # TODO - get all constants from dictionary
        Re = param_dict["Re"]
        Pr = param_dict["Pr"]
        new{N}(Re, Pr)
    end
end
is_second_order(app::NavierStokes) = true
nstates(app::NavierStokes{N}) where {N} = N+2
"""
    flux_c!(flux, app::NavierStokes{N}, u)
Computes the convective (inviscid) flux for the Navier-Stokes equations:
f_c([ρ,ρu,ρE]) = [ρu, ρ u⊗u'+pI, (ρE+p)u] ∈ ℜ^(nstates × N)
"""
function flux_c!(flux, app::NavierStokes{N}, u) where {N}
    ρ = u[1]
    ρu = @view u[2:1+N]
    ρE = u[2+N]
    γ = 7/5
    p = (γ-1)*(ρE-1/2*(ρu'*ρu)/ρ)
    flux[1,:] .= ρu
    flux[2:1+N,:] .= (ρu*ρu')./ρ
    for i = 1:N
        flux[1+i,i] += p
    end
    flux[2+N,:] .= ρu*(ρE+p)./ρ
    flux
end
"""
    flux_d!(flux, app::NavierStokes{N}, u, Du)
Computes the diffusive (viscous) flux for Navier-Stokes equations:
f_d([ρ,ρu,ρE]) = [0, τ, τu-q] ∈ ℜ^(nstates × N)
"""
function flux_d!(flux, app::NavierStokes{N}, state, Dstate) where {N}
    # TODO: Time with @view in different spots
    ρ = state[1]
    ρu = @view state[2:1+N]
    ρE = state[2+N]
    ∇ρ = @view Dstate[1,:]
    ∇ρu = @view Dstate[2:1+N,:]
    ∇ρE = @view Dstate[2+N,:]
    γ = 7/5
    u = ρu/ρ
    ∇u = (∇ρu-u*∇ρ')/ρ
    div_u = sum(∇u[LinearAlgebra.diagind(∇u)])
    ∇E = (∇ρE-(ρE/ρ)*∇ρ)/ρ

    flux[1,:] .= 0.0
    # Compute viscous stress tensor τ in-place
    τ = @view flux[2:1+N,:]
    τ .= ∇u + ∇u'
    for i = 1:N
        τ[i,i] -= 2/3*div_u
    end
    τ ./= app.Re
    # Compute heat flux vector in-place
    flux[2+N,:] .= γ/app.Pr/app.Re*(∇E-∇u'*u)
    flux[2+N,:] .+= τ*u
    flux
end
"""
    numerical_flux_c!(flux, app::NavierStokes{N}, uK, uN, normal_k) where {N}
TODO: Roe numerical flux
"""
function numerical_flux_c!(flux, app::NavierStokes{N}, uK, uN, normal_k) where {N}
    # TODO
    error("Not yet programmed")
end
