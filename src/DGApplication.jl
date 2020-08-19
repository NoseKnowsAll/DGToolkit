import LinearAlgebra

"""
    abstract type DGApplication{N}
Supertype for applications (PDEs) that specific PDEs can extend. Generalizes
the equation `u_t + ∇⋅f_c(u) + ∇⋅f_d(u,∇u) = s(t,x) where u = u(t,x)`. N is the
dimension of the PDE, aka size of the vector x.
"""
abstract type DGApplication{N} end

"""
    is_second_order(app::DGApplication)
Returns whether the PDE is 2nd order. Default: false => f_d(u,∇u) = 0
"""
is_second_order(app::DGApplication) = false
"""
    nstates(app::DGApplication)
Returns the number of state variables for the given PDE. Default: 1
"""
nstates(app::DGApplication) = 1
"""
    flux_c!(flux, app::DGApplication, u)
Computes the 1st order, or convective, flux f_c(u) for the given application.
Default: f_c(u) = 0.0
"""
flux_c!(flux, app::DGApplication{N}, u) where {N} = (flux .= zeros(typeof(u[1]),size(u,1),N))
"""
    flux_d!(flux, app::DGApplication, u)
Computes the 2nd order, or diffusive, flux f_d(u,∇u) for the given application.
Default: f_d(u,∇u) = 0.0
"""
flux_d!(flux, app::DGApplication{N}, u, Du) where {N} = (flux .= zeros(typeof(u[1]),size(u,1),N))
"""
    numerical_flux_c!(flux, app::DGApplication, uK, uN, normal_k)
Given the state vector at this element (uK) and its neighbor (uN), compute the
convection numerical flux function for this PDE dotted with the normal, and
store the results in flux. Default: f(uK,uN)*n = upwinding with positive speed
"""
function numerical_flux_c!(flux, app::DGApplication{N}, uK, uN, normal_k) where {N}
    @assert N == length(normal_k)
    flux_k = Array{Float64,2}(undef, nstates(app), N)
    flux_c!(flux_k, app, uK)
    flux_n = Array{Float64,2}(undef, nstates(app), N)
    flux_c!(flux_n, app, uN)
    flux .= zero(typeof(uK[1]), nstates(app))
    for l=1:N
        @views flux .+= (normal_k[l] > 0.0 ? flux_k[:,l] : flux_n[:,l]).*normal_k[l]
    end
end
"""
    source!(s_val, app::DGApplication, t=0.0, x=0.0)
Computes the source(t,x) of the PDE. Default: source(t,x) = 0.0
"""
source!(s_val, app::DGApplication, t=0.0, x=0.0) = (s_val .= zeros(typeof(u[1]),1))


"""
    struct Convection{N} <: DGApplication{N}
Defines first-order linear convection equation u_t + c⋅∇u = 0
"""
struct Convection{N} <: DGApplication{N}
    c
    Convection{N}(c) where {N} = new{N}(c)
    function Convection{N}(param_dict::Dict{String,Any}) where {N}
        c = param_dict["c"]
        new{N}(c)
    end
end

"""
    flux_c!(flux, app::Convection, u)
Computes the convective flux for convection equation: f_c(u) = cu
"""
flux_c!(flux, app::Convection, u) = (flux .= app.c.*u)


"""
    struct ConvectionDiffusion{N} <: DGApplication{N}
Defines linear convection-diffusion equation u_t + c⋅∇u - dΔu = 0
Must initialize struct with constants c,d
"""
struct ConvectionDiffusion{N} <: DGApplication{N}
    c
    d
    ConvectionDiffusion{N}(c,d) where {N} = new{N}(c,d)
    function ConvectionDiffusion{N}(param_dict::Dict{String,Any}) where {N}
        c = param_dict["c"]
        d = param_dict["d"]
        new{N}(c,d)
    end
end
is_second_order(app::ConvectionDiffusion) = true
"""
    flux_c!(flux, app::ConvectionDiffusion, u)
Computes the convective flux for convection-diffusion equation: f_c(u) = cu
"""
flux_c!(flux, app::ConvectionDiffusion, u) = (flux .= app.c.*u)
"""
    flux_d!(flux, app::ConvectionDiffusion, u, Du)
Computes the diffusion flux for convection-diffusion equation: f_d(u,∇u)=d∇u
"""
flux_d!(flux, app::ConvectionDiffusion, u, Du) where {N} = (flux .= app.d.*Du)


"""
    struct NavierStokes{N} <: DGApplication{N}
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
struct NavierStokes{N} <: DGApplication{N}
    # Constants TBD
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
    @views div_u = sum(∇u[LinearAlgebra.diagind(∇u)])
    ∇E = (∇ρE-(ρE/ρ)*∇ρ)/ρ

    flux[1,:] .= 0.0
    # Compute viscous stress tensor τ in-place
    τ = @view flux[2:1+N,:]
    τ .= ∇u + ∇u'
    for i = 1:N
        τ[i,i] -= 2/3*div_u
    end
    τ ./= app.Re
    # Compute heat flux
    q = -γ/app.Pr/app.Re*(∇E-∇u'*u)
    flux[2+N,:] .= τ*u-q
    flux
end
"""
    numerical_flux_c!(flux, app::NavierStokes{N}, uK, uN, normal_k) where {N}
TODO: Roe numerical flux
"""
function numerical_flux_c!(flux, app::NavierStokes{N}, uK, uN, normal_k) where {N}
    # TODO
end
