module DGApplication

import LinearAlgebra

# Types
export Application
export Convection, ConvectionDiffusion, NavierStokes, ElasticWave
# Functions
export is_second_order, nstates
export flux_c!, flux_d!
export numerical_flux_c!, boundary_flux_c!
export source!


"""
    abstract type Application{N}
Supertype for applications (PDEs) that specific PDEs can extend. Generalizes
the equation `u_t + ∇⋅f_c(u) + ∇⋅f_d(u,∇u) = s(t,x) where u = u(t,x)`. N is the
dimension of the PDE, aka size of the vector x.
"""
abstract type Application{N} end

"""
    is_second_order(app::Application)
Returns whether the PDE is 2nd order. Default: false => f_d(u,∇u) = 0
"""
is_second_order(app::Application) = false
"""
    nstates(app::Application)
Returns the number of state variables for the given PDE. Default: 1
"""
nstates(app::Application) = 1
"""
    flux_c!(flux, app::Application, u)
Computes the 1st order, or convective, flux f_c(u) for the given application.
Default: f_c(u) = 0.0
"""
flux_c!(flux, app::Application{N}, u) where {N} = (flux .= zeros(typeof(u[1]),nstates(app),N))
"""
    flux_d!(flux, app::Application, u)
Computes the 2nd order, or diffusive, flux f_d(u,∇u) for the given application.
Default: f_d(u,∇u) = 0.0
"""
flux_d!(flux, app::Application{N}, u, Du) where {N} = (flux .= zeros(typeof(u[1]),nstates(app),N))
"""
    numerical_flux_c!(flux, app::Application, uK, uN, normal_k)
Given the state vector at this element (uK) and its neighbor (uN), compute the
convection numerical flux function for this PDE dotted with the normal, and
store the results in flux. Default: f(uK,uN)*n = upwinding with positive speed
"""
function numerical_flux_c!(flux, app::Application{N}, uK, uN, normal_k) where {N}
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
    boundary_flux_c!(flux, app::Application, u, bc, normal_k)
Given the state vector u, compute the boundary condition numerical flux function
for this PDE dotted with the normal, and store the results in flux.
Default: f(u)*n = 0.0 (Homogeneous Dirichlet BC)
"""
function boundary_flux_c!(flux, app::Application, u, bc, normal_k)
    flux .= zeros(typeof(u[1]),nstates(app))
end
"""
    source!(s_val, app::Application, t=0.0, x=0.0)
Computes the source(t,x) of the PDE. Default: source(t,x) = 0.0
"""
source!(s_val, app::Application, t=0.0, x=0.0) = (s_val .= zeros(nstates(app)))


"""
    struct Convection{N} <: Application{N}
Defines first-order linear convection equation u_t + c⋅∇u = 0
"""
struct Convection{N} <: Application{N}
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
    struct ConvectionDiffusion{N} <: Application{N}
Defines linear convection-diffusion equation u_t + c⋅∇u - dΔu = 0
Must initialize struct with constants c,d
"""
struct ConvectionDiffusion{N} <: Application{N}
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


"""
    struct ElasticWave{N} <: Application{N}
Defines the elastic wave equation in conservation form:
∂E/∂t + ∇⋅(-1/2(u⊗I+I⊗u)) = 0
∂(ρu)/∂t + ∇⋅(-C:E) = f
where ρ is the density, u is the velocity, E is the strain tensor, f is a
forcing function or source term, and I is the N × N identity tensor. Defining
λ,μ to be the Lame parameters, in isotropic media, stress tensor S=CE=2μE+λtr(E)I
"""
struct ElasticWave{N} <: Application{N}
    " Pressure wave velocity (actually speed)"
    vp
    " Shear wave velocity (actually speed) "
    vs
    " ρ = Density "
    ρ
    " λ = 1st Lame parameter "
    λ
    " μ = 2nd Lame parameter "
    μ
    ElasticWave{N}(vp,vs,ρ,λ,μ) where {N} = new{N}(vp,vs,ρ,λ,μ)
    function ElasticWave{N}(param_dict::Dict) where {N}
        vp = param_dict["vp"]
        vs = param_dict["vs"]
        ρ = param_dict["ρ"]
        λ = param_dict["λ"]
        μ = param_dict["μ"]
        new{N}(vp,vs,ρ,λ,μ)
    end
end
nstates(app::ElasticWave{N}) where {N} = Int64(N*(N+1)/2)+N
"""
    flux_c!(flux, app::ElasticWave{N}, u)
Computes the convective flux for the Elastic wave equation:
f_c([E,ρu]) = [-1/2(u⊗I+I⊗u), -CE] ∈ ℜ^(nstates × N)
"""
function flux_c!(flux, app::ElasticWave{N}, u) where {N}
    nstrains = N*(N+1)/2
    # Remember - we are using Voigt notation for strain tensor
    Ediag = @view u[1:N]
    Eupp  = @view u[N+1:nstrains]
    vel = u[nstrains+1:end]./app.ρ
    # Compute -1/2(vel⊗I + I⊗vel)
    flux .= 0.0
    for i = 1:N # diagonal of strain tensor
        flux[i,i] = -vel[i]
    end
    offset = 1
    for i = N:-1:2 # upper-triangular portion of strain tensor
        for j = i-1:-1:1
            flux[N+offset,i] = -1/2*vel[j]
            flux[N+offset,j] = -1/2*vel[i]
            offset += 1
        end
    end
    # Compute -CE = -2μE - λtr(E)I
    trEλ = app.λ*sum(Ediag)
    for i = 1:N
        flux[nstrains+i,i] = -2*app.μ*Ediag[i] - trEλ
    end
    offset = 1
    for i = N:-1:2
        for j = i-1:-1:1
            flux[nstrains+i,j] = -2*app.μ*Eupp[offset]
            flux[nstrains+j,i] = -2*app.μ*Eupp[offset]
            offset += 1
        end
    end
    flux
end
"""
    numerical_flux_c!(flux, app::ElasticWave, uK, uN, normal_k)
Given the state vector at this element (uK) and its neighbor (uN), compute the
Lax-Friedrichs flux function of this PDE dotted with the normal, and
store the results in flux.
"""
function numerical_flux_c!(flux, app::ElasticWave{N}, uK, uN, normal_k) where {N}
    @assert N == length(normal_k)
    flux_k = Array{Float64,2}(undef, nstates(app), N)
    flux_c!(flux_k, app, uK)
    flux_n = Array{Float64,2}(undef, nstates(app), N)
    flux_c!(flux_n, app, uN)
    # Max speed is pressure wave speed
    C = app.vp
    flux .= (flux_k+flux_n)*normal_k./2.0 - (C/2.0)*(uN-uK)
end
"""
    boundary_flux_c!(flux, app::ElasticWave, u, bc, normal_k)
Free surface boundary condition (v=0) and absorbing boundary condition (Z*v)
for elastic wave equation
"""
function boundary_flux_c!(flux, app::ElasticWave{N}, u, bc, normal_k) where {N}
    nstrains = N*(N+1)/2
    vel = u[nstrains+1:end]./app.ρ
    flux .= 0.0
    # Remember - we are using Voigt notation for strain tensor
    # Compute -1/2(vel⊗I + I⊗vel)*n for Strain fluxes
    for i = 1:N # diagonal of strain tensor
        flux[i] = -vel[i]*normal_k[i]
    end
    offset = 1
    for i = N:-1:2 # upper-triangular portion of strain tensor
        for j = i-1:-1:1
            flux[N+offset] += -1/2*vel[j]*normal_k[i]
            flux[N+offset] += -1/2*vel[i]*normal_k[j]
            offset += 1
        end
    end
    if bc == -1 # Free surface BC
        # flux[nstrains+1:end] = 0.0
    elseif bc == -2 # Absorbing/non-reflecting BC
        # flux[nstrains+1:end] = ρ (vp n⊗n + vs (I-n⊗n))*vel
        Z = app.ρ*(app.vp*normal_k*normal_k'+app.vs*(diagm(ones(N))-normal_k*normal_k'))
        flux[nstrains+1:end] .= Z*vel
    else
        error("FATAL: Boundary condition not a valid choice!")
    end
end
"""
    source!(s_val, app::ElasticWave, t=0.0, x=0.0)
Computes the source(t,x) of the elastic wave equation:
source(t,x) = [0, ρ*wave(t)*g(x)]
"""
function source!(s_val, app::ElasticWave{N}, t=0.0, x=0.0) where {N}
    nstrains = N*(N+1)/2
    pts_per_lambda_min = 4
    mesh_max_dx = 0.1 # TODO: Does this need to be an app param?
    max_freq = app.vs/(pts_per_lambda_min*mesh_max_dx)
    freq = 0.9*max_freq
    wave_amp = cos(2*π*freq*t)
    s_val[1:nstrains] .= 0.0
    # TODO: g(x) := 1 => source wave active at all locations
    s_val[nstrains+1:end] .= app.ρ*wave_amp*1
    s_val
end

end # Module DGApplication
