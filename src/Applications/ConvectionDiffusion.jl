export ConvectionDiffusion

"""
    struct ConvectionDiffusion{N} <: Application{N}
Defines linear convection-diffusion equation u_t + c⋅∇u - dΔu = 0
Default Dirichlet BC (element neighbor == -1): u = 0.0
Default Neumann BC (element neighbor == -2): ∇u⋅n = 0.0
Must initialize struct with constants c,d
"""
struct ConvectionDiffusion{N} <: Application{N}
    c    # Convection constant
    d    # Diffusion constant
    bc_d # Dirichlet BC
    bc_n # Neumann BC
    ConvectionDiffusion{N}(c,d,bc_d=0.0,bc_n=0.0) where {N} = new{N}(c,d,bc_d,bc_n)
    function ConvectionDiffusion{N}(param_dict::Dict{String,Any}) where {N}
        c = param_dict["c"]
        d = param_dict["d"]
        bc_d = get!(param_dict, "bc_d", 0.0)
        bc_n = get!(param_dict, "bc_n", 0.0)
        new{N}(c,d, bc_d,bc_n)
    end
end
is_second_order(app::ConvectionDiffusion) = true
"""
    flux_c!(flux, app::ConvectionDiffusion, u)
Computes the convective flux for convection-diffusion equation: f_c(u) = cu
"""
flux_c!(flux, app::ConvectionDiffusion, u) = (flux .= u*app.c')
"""
    flux_d!(flux, app::ConvectionDiffusion, u, Du)
Computes the diffusion flux for convection-diffusion equation: f_d(u,∇u)=d∇u
"""
flux_d!(flux, app::ConvectionDiffusion, u, Du) where {N} = (flux .= app.d.*Du)

function boundary_flux_c!(flux, app::ConvectionDiffusion, u, bc, normal_k)
    flux_k = similar(u)
    if bc == -1 # Dirichlet BC: u = bc_d
        boundary_u = fill(app.bc_d, size(u))
        flux_c!(flux_k, app, boundary_u)
    elseif bc == -2 # Neumann BC: ∇u⋅n = bc_n
        flux_c!(flux_k, app, u)
    end
    flux .= flux_k.*normal_k
end

function boundary_flux_d!(flux, app::ConvectionDiffusion, u,Du, bc, normal_k)
    fluxes = similar(Du)
    if bc == -1 # Dirichlet BC: u = bc_d
        boundary_u = fill(app.bc_d,size(u))
        flux_d!(fluxes, app, boundary_u, Du)
    elseif bc == -2 # Neumann BC: ∇u⋅n = bc_n
        boundary_Du = fill(app.bc_n,size(Du))
        flux_d!(fluxes, app, u, boundary_Du)
    end
    flux .= fluxes.*normal_k
end

function boundary_flux_l!(flux, app::ConvectionDiffusion, u, bc, normal_k)
    flux_k = similar(u)
    if bc == -1 # Dirichlet BC: u = bc_d
        boundary_u = fill(app.bc_d, size(u))
        flux_l!(flux_k, app, boundary_u)
    elseif bc == -2 # Neumann BC: ∇u⋅n = bc_n
        flux_l!(flux_k, app, u)
    end
    flux .= flux_k.*normal_k
end
