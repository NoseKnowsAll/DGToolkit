module DGApplication

import LinearAlgebra

# Types
export Application
# Functions
export is_second_order, nstates
export flux_c!, flux_d!, flux_l!
export numerical_flux_c!, boundary_flux_c!
export numerical_flux_d!, boundary_flux_d!
export numerical_flux_l!, boundary_flux_l!
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
    flux_l!(flux, app::Application, u)
Computes the local DG flux function for the given application. For CDG, f_l(u)=u
Default: f_l(u) = u
"""
flux_l!(flux, app::Application{N}, u) where {N} = (flux .= deepcopy(u))
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
    flux .= zeros(typeof(uK[1]), nstates(app))
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
    function numerical_flux_l!(flux, app::Application, uK, uN, switch, normal_k)
Compute the numerical local DG flux function dotted with n for this PDE. Note
that there is a different local DG flux depending on the gradient directions.
Therefore, normal_k is actually only the normal in the direction we care about.
Default: f(u)*n = f(uK)*n or f(uN)*n based on mesh's Local DG switch
"""
function numerical_flux_l!(flux, app::Application{N}, uK, uN, switch, normal_k) where {N}
    flux_k = similar(uK)
    flux_l!(flux_k, app, uK)
    flux_n = similar(uN)
    flux_l!(flux_n, app, uN)
    flux .= (switch ? flux_k : flux_n)*normal_k
end
"""
    function boundary_flux_l!(flux, app::Application{N}, u, bc, normal_k)
Compute the numerical local DG flux function dotted with n for this PDE on the
boundary of the domain. Note that there is a different local DG flux depending
on the gradient directions. Therefore normal_k is actually only the normal in
the direction we care about. Must override function for different BCs than
bc==-1 => Homogeneous Dirichlet and bc==-2 => Homogeneous Neumann
Default: f(u)*n = given bc*n for Dirichlet BC and f(u)*n for Neumann BC
"""
function boundary_flux_l!(flux, app::Application{N}, u, bc, normal_k) where {N}
    flux_k = similar(u)
    if bc == -1 # Homogeneous Dirichlet BC
        fill!(flux_k, 0.0)
    elseif bc == -2 # Homogeneous Neumann BC
        flux_l!(flux_k, app, u)
    else
        flux_l!(flux_k, app, u)
        error("Unknown default BC - must override operator!")
    end
    flux .= flux_k*normal_k
end
"""
    function numerical_flux_d!(flux, app::Application, uK,uN, DuK,DuN, switch, normal_k)
Compute the numerical diffusive (viscous) flux dotted with n for this PDE.
Default: f(uK,uN,DuK,DuN) = -f_d(uK, !ldg_switch(DuK,DuN))*n
"""
function numerical_flux_d!(flux, app::Application{N}, uK,uN, DuK,DuN, switch, normal_k) where {N}
    fluxes = similar(DuK)
    if !switch # Negative ldg switch for choosing DuK or DuN
        flux_d!(fluxes, app, uK, DuK)
    else
        flux_d!(fluxes, app, uK, DuN)
    end
    # flux = -fluxes*n
    LinearAlgebra.mul!(flux, fluxes, normal_k, -1.0, 0.0)
end
"""
    function boundary_flux_d!(flux, app::Application{N}, u,Du,bc, normal_k)
Compute the numerical diffusive (viscous) flux dotted with n for this PDE at a
boundary.
Default: f(u,Du)*n = 0.0 (Homogeneous Neumann BC)
"""
function boundary_flux_d!(flux, app::Application{N}, u,Du,bc, normal_k)
    flux .= zeros(typeof(Du[1]),size(Du)...)
end
"""
    source!(s_val, app::Application, t=0.0, x=0.0)
Computes the source(t,x) of the PDE. Default: source(t,x) = 0.0
"""
source!(s_val, app::Application, t=0.0, x=0.0) = (s_val .= zeros(nstates(app)))

include("Applications\\Convection.jl")
include("Applications\\ConvectionDiffusion.jl")
include("Applications\\NavierStokes.jl")
include("Applications\\ElasticWave.jl")

end # Module DGApplication
