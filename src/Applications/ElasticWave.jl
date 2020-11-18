export ElasticWave

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
