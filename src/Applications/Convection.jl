export Convection

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
flux_c!(flux, app::Convection, u) = (flux .= u*app.c')
