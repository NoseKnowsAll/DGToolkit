export Convection

"""
    struct Convection{N} <: Application{N}
Defines first-order linear convection equation u_t + c⋅∇u = 0
with optional Dirichlet boundary condition: u(boundary) = bc_d
"""
struct Convection{N} <: Application{N}
    c    # Convection constant
    bc_d # Dirichlet BC
    Convection{N}(c,bc_d=0.0,bc_n=0.0) where {N} = new{N}(c,bc_d,bc_n)
    function Convection{N}(param_dict::Dict{String,Any}) where {N}
        c = param_dict["c"]
        bc_d = get!(param_dict, "bc_d", 0.0)
        new{N}(c, bc_d, bc_n)
    end
end

"""
    flux_c!(flux, app::Convection, u)
Computes the convective flux for convection equation: f_c(u) = cu
"""
flux_c!(flux, app::Convection, u) = (flux .= u*app.c')

function boundary_flux_c!(flux, app::Convection, u, bc, normal_k)
    flux_k = similar(u)
    if bc == -1 # Dirichlet BC: u = bc_d
        boundary_u = fill(app.bc_d, size(u))
        flux_c!(flux_k, app, boundary_u)
    elseif bc == -2 # Neumann BC: ∇u⋅n = 0
        flux_c!(flux_k, app, u)
    end
    flux .= flux_k.*normal_k
end
