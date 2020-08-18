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
    flux_k = Array{Float64,2}(undef, size(uK,1), N)
    flux_c!(flux_k, app, uK)
    flux_n = Array{Float64,2}(undef, size(uN,1), N)
    flux_c!(flux_n, app, uN)
    flux .= zero(typeof(uK[1]), size(uK,1))
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
Computes the diffusion flux for convection-diffusion equation: f_d(u,∇u)=-d∇u
"""
flux_d!(flux, app::ConvectionDiffusion, u, Du) where {N} = (flux .= -app.d.*Du)
