# All FiniteElement code dealing with implicit time integration.
# Directly included by FiniteElement.jl and inside module FiniteElement

"""
    function jac_assemble(solver, t)
Assembles the Jacobian matrix d(dg_rhs)/du at the current time step. Assumes
dg_rhs!() has just been called at this time step/stage and solver.u_interpV and
solver.u_interpF are precomputed.
"""
function jac_assemble(solver, t)
    if is_second_order(solver.app)
        # d∇u/du = (I⊗M) \ (d F_l(u)/du - Gs)
        compute_dgrad_du!(solver.dgrad_du, solver)
        # JD
    end
    jac_convect_volume(JD, solver)
    jac_convect_flux(JD, solver)
end
"""
    function compute_dgrad_du!(dgrad_du, solver)
Assembles the Jacobian matrix d(∇u)/du into dgrad_du. Computed as
d∇u/du = (I⊗M) \\ (d F_l(u)/du - Gs)
"""
function compute_dgrad_du!(dgrad_du, solver)
    for l=1:Geometry.DIM
        # TODO: Remap Gs to matrix array: 1 for each dimension
        dgrad_du .= -solver.Gels
        dlocal_flux_du!(dgrad_du, solver)
        # dgrad_du = M\dgrad_du
        ldiv!(solver.Mel, dgrad_du[l])
    end
end
"""
    function dlocal_flux_du!(dgrad_du, solver)
Updates dgrad_du += d(F_l(u))/du
"""
function dlocal_flux_du!(dgrad_du, solver)
    # TODO
end
"""
    function jac_convect_volume(solver)
Computes the jacobian of the convect_volume! function wrt u and adds it to JD,
the diagonal block matrix of the Jacobian so far.
"""
function jac_convect_volume(JD, solver)

end
