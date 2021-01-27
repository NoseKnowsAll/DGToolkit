"""
    Module Solver
Information about finite element spaces and DG solver
"""
module FiniteElement

import Basis1D
import Basis2D
import Geometry
using DGApplication
using LinearAlgebra
using FELinearAlgebra

export Solver
export dg_rhs!
export nstates, nelements, dofsV, dofsF, nQV, nQF

"""
    struct Solver
All DG-specific spatial discretization/solver information
"""
struct Solver
    " Mesh on which FE space is defined "
    mesh::Geometry.Mesh
    " Application defining the PDE itself we are spatial discretizing "
    app::Application
    " Order of DG method "
    order::Int
    " Reference nodes in 1D. N-D version can be formed by kronecker product "
    x1D
    " Gauss quadrature points in 1D. N-D version can be formed by kronecker product "
    xQ1D
    " Gauss quadrature weights in ND, for use within volume "
    wQV
    " Gauss quadrature weights in (N-1)D, for use along faces "
    wQF
    ###########
    # TODO: should these matrices possibly be stored in a more abstract fashion?
    ###########
    " Global element mass matrix, defined as ∫ Φ_i Φ_j dx "
    Mel
    " Global element gradient matrix, defined as ∫ ∇_lΦ_i Φ_j dx "
    Gels
    " Global element stiffness matrix, defined as ∫ ∇Φ_i ⋅ ∇Φ_j dx"
    Sel
    ###########
    # End TODO
    ###########
    " Interpolation matrix from element nodes to element quadrature points "
    InterpV
    " Interpolation matrix from face nodes to face quadrature points "
    InterpF
    """
    Gradient of basis functions interpolating from element nodes to
    element quadrature points. Can alternatively be viewed as local element
    gradient matrix K, defined as ∇_lΦ_i
    """
    dInterpV
    """
    Gradient of basis functions interpolating from face nodes to face
    quadrature points. Can alternatively be viewed as a local element face
    gradient matrix, defined as ∇_lΦ_i along faces
    """
    dInterpF
    " Interpolation matrix from bilinear map Tk points to element nodes "
    InterpTk
    " Interpolation matrix from bilinear map Tk points to element quadrature points "
    InterpTkQ

    #### Work arrays
    " Solution vector stored on volume quadrature points "
    u_interpV
    " Solution vector stored on face quadrature points "
    u_interpF
    " Array of `SolutionVector`, storing ∇u in each dimension "
    ∇u
    " Single `SolutionVector`, storing ∇u with fastest dimension = dim*nodes"
    ∇u_compute
    " Gradient ∇u stored on volume quadrature points "
    ∇u_interpV
    " Gradient ∇u stored on face quadrature points "
    ∇u_interpF

    function Solver(order::Int, mesh::Geometry.Mesh, app::Application)
        # Initialize node, quadrature, and interpolation matrix information
        (x1D,xQ1D,wQV,wQF,xTk1D,InterpV,InterpF,
            dInterpV,dInterpF,InterpTk,InterpTkQ) =
            precompute_interpolation_matrices(order)
        # Update Mesh to know about solver information
        Geometry.setup_nodes!(mesh,InterpTk,order,compute_global=true)
        Geometry.setup_quads!(mesh,InterpTkQ)
        Geometry.precompute_jacobians!(mesh,xTk1D,xQ1D)
        if is_second_order(app)
            Geometry.precompute_ldg_switch!(mesh)
        end
        # Initialize work arrays
        ns = DGApplication.nstates(app)
        ne = mesh.n_elements
        #u = SolutionVector{Float64}(size(InterpV,2), ns, ne)
        u_interpV = Array{Float64,3}(undef, size(InterpV,1),ns,ne)
        u_interpF = Array{Float64,4}(undef, size(InterpF,1),ns,Geometry.N_FACES,ne)
        if is_second_order(app)
            ∇u = [SolutionVector{Float64}(size(InterpV,2),ns,ne) for i=1:Geometry.DIM]
            ∇u_compute = SolutionVector{Float64}(Geometry.DIM*size(InterpV,2),ns,ne)
            ∇u_interpV = Array{Float64,4}(undef, size(InterpV,1),ns,ne,Geometry.DIM)
            ∇u_interpF = Array{Float64,5}(undef, size(InterpF,1),ns,Geometry.N_FACES,ne,Geometry.DIM)
        else
            ∇u = []
            ∇u_compute = []
            ∇u_interpV = []
            ∇u_interpF = []
        end
        # Initialize local matrices
        Locals = precompute_local_matrices(order,mesh,app,wQV,InterpV,InterpF,dInterpV,dInterpF)

        return new(mesh,app,order, x1D,xQ1D,wQV,wQF, Locals...,
            InterpV,InterpF,dInterpV,dInterpF,InterpTk,InterpTkQ,
            u_interpV,u_interpF, ∇u,∇u_compute,∇u_interpV,∇u_interpF)
    end

end

" Number of states in state vector "
DGApplication.nstates(solver::Solver) = DGApplication.nstates(solver.app)
" Number of elements in mesh "
nelements(solver::Solver) = solver.mesh.n_elements
" Degrees of freedom per element "
dofsV(solver::Solver) = size(solver.InterpV,2)
" Degrees of freedom per element face "
dofsF(solver::Solver) = size(solver.InterpF,2)
" Number of quadrature points per element "
nQV(solver::Solver) = size(solver.InterpV,1)
" Number of quadrature points per element face "
nQF(solver::Solver) = size(solver.InterpF,1)

"""
    precompute_interpolation_matrices(order)
Precompute all interpolation matrices used by DG method.
"""
function precompute_interpolation_matrices(order)
    # Reference nodes
    x1D = Basis1D.chebyshev(order)
    # Gauss quadrature points
    (xQ1D,wQ1D) = Basis1D.gauss_quad(2*order)
    wQV = kron(wQ1D,wQ1D)
    wQF = deepcopy(wQ1D)

    # Volume/Element interpolation
    InterpV = Basis2D.interpolation_matrix2D((x1D,x1D),(xQ1D,xQ1D))
    # Face interpolation
    InterpF = Basis1D.interpolation_matrix1D(x1D, xQ1D)

    # Local gradient matrix = gradient of reference bases on the quad points
    dInterpV = Basis2D.dPhi2D((x1D,x1D),(xQ1D,xQ1D))
    # Local face gradient matrix
    dInterpF = Basis1D.dPhi1D(x1D, xQ1D)

    # Interpolation from points on which bilinear_mapping is defined
    xTk1D = Basis1D.chebyshev(1)
    InterpTk = Basis2D.interpolation_matrix2D((xTk1D,xTk1D),(x1D,x1D))
    InterpTkQ = Basis2D.interpolation_matrix2D((xTk1D,xTk1D),(xQ1D,xQ1D))

    return (x1D,xQ1D,wQV,wQF,xTk1D,InterpV,InterpF,dInterpV,dInterpF,InterpTk,InterpTkQ)
end

"""
    precompute_local_matrices()
Precompute all element-local matrices used by DG method.
"""
function precompute_local_matrices(order, mesh::Geometry.Mesh, app::Application,
    wQV,InterpV,InterpF,dInterpV,dInterpF)
    dofsV = size(InterpV,2)
    dofsF = size(InterpF,2)
    nQV = size(InterpV,1)
    nQF = size(InterpF,1)

    # Mass matrix = ∫ Φ_i Φ_j dx
    Mdata = Array{Float64,4}(undef, dofsV,dofsV,1,mesh.n_elements)
    for iK = 1:mesh.n_elements
        # TODO: Eventually this should be done in matrix-free approach
        scaling = wQV.*vec(mesh.Jkdet[:,iK])
        Mdata[:,:,1,iK] = InterpV'*Diagonal(scaling)*InterpV
    end
    Mel = LocalMatrix(Mdata)

    # Gradient matrix = ∫ ∇ Φ_i Φ_j dx
    if is_second_order(app)
        Gdata = zeros(Float64, Geometry.DIM,dofsV,dofsV,1,mesh.n_elements)
        for iK = 1:mesh.n_elements
            # TODO: Eventually this should be done in matrix-free approach
            Jks = @view mesh.Jk[:,:,:,iK]
            # op_q = wQ * J^{-1}*det(J) == wQ * adj(J)
            op_q = Array{Float64,3}(undef, size(Jks))
            op_q[1,1,:] = wQV.* Jks[2,2,:]
            op_q[1,2,:] = wQV.*-Jks[1,2,:]
            op_q[2,1,:] = wQV.*-Jks[2,1,:]
            op_q[2,2,:] = wQV.* Jks[1,1,:]
            @views for iQ = 1:nQV
                grad = op_q[:,:,iQ]*(dInterpV[iQ,:,:]')
                Gdata[1,:,:,1,iK] .+= InterpV[iQ,:]'*grad[1,:]
                Gdata[2,:,:,1,iK] .+= InterpV[iQ,:]'*grad[2,:]
            end
        end
        Gdata = reshape(Gdata,Geometry.DIM*dofsV,dofsV,1,mesh.n_elements)
    else
        Gdata = zeros(Float64, 0,0,0,0)
    end
    Gels = LocalMatrix(Gdata)

    # Stiffness matrix = ∫ ∇Φ_i ⋅ ∇Φ_j dx
    Sdata = zeros(Float64, dofsV,dofsV,1,mesh.n_elements)
    for iK = 1:mesh.n_elements
        # TODO: Eventually this should be done in matrix-free approach
        Jks = @view mesh.Jk[:,:,:,iK]
        # op_q = wQ * J^{-1}*det(J)*J^{-T} == wQ/det(J)*adj(J)*adj(J)^T
        op_q = Array{Float64,3}(undef, size(Jks))
        op_q[1,1,:] =  wQV./mesh.Jkdet[:,iK].*(Jks[1,2,:].^2+Jks[2,2,:].^2)
        op_q[1,2,:] = -wQV./mesh.Jkdet[:,iK].*(Jks[1,2,:].*Jks[1,1,:]+Jks[2,2,:].*Jks[2,1,:])
        op_q[2,1,:] = op_q[1,2,:]
        op_q[2,2,:] =  wQV./mesh.Jkdet[:,iK].*(Jks[1,1,:].^2+Jks[2,1,:].^2)
        @views for iQ = 1:nQV
            grad = op_q[:,:,iQ]*(dInterpV[iQ,:,:]')
            Sdata[:,:,1,iK] .+= dInterpV[iQ,:,:]*grad
        end
    end
    Sel = LocalMatrix(Sdata)

    return (Mel, Gels, Sel)
end


"""
    dg_rhs!(du, u_curr, all_param, t)
Given u_curr, evaluate the RHS of the system of ODE: u_t = f(u_curr, t)
where f represents the spatial discretization of our application
Note that all_param is a tuple containing (solver,) TODO: what else?
this is equivalent to du = Mel \\ (S+K_c(u)-F_c(u)) in the case of 1st order apps or
∇u = Mel \\ (-Gels*u + F_l(u)),
du = Mel \\ ( S+K_c(u)+K_d(u,∇u)-F_c(u)-F_d(u,∇u) ) in the case of 2nd order apps.
"""
function dg_rhs!(du, u_curr, all_param, t)
    (solver,) = all_param # TODO: TBD - what else?

    interpolate!(solver, u_curr, solver.u_interpV, solver.u_interpF)
    if is_second_order(solver.app)
        # ∇u = -Gs*u
        mul!(solver.∇u_compute,solver.Gels,u_curr,-1.0,0.0)
        convert_to_dim_array!(solver.∇u, solver.∇u_compute)
        # ∇u += F_l(u)
        local_flux!(solver)

        # ∇u = (I⊗M)\∇u
        for l=1:Geometry.DIM
            ldiv!(solver.Mel, solver.∇u[l])
            @views interpolate!(solver, solver.∇u[l],
                solver.∇u_interpV[:,:,:,l], solver.∇u_interpF[:,:,:,:,l])
        end

    end
    # du = s
    source_term!(du, solver, t)
    # du -= F_c(u)
    convect_flux!(du, solver)
    # du += K_c(u)
    convect_volume!(du, solver)
    if is_second_order(solver.app)
        # du -= F_d(u,∇u)
        diffusive_flux!(du, solver)
        # du += K_d(u,∇u)
        diffusive_volume!(du, solver)
    end

    # du = M\du
    ldiv!(solver.Mel, du)
    # Mass matrix might be addable directly to ODE - double check documentation

end

"""
    interpolate!(solver::Solver, curr::SolutionVector, to_interpV, to_interpF)
Interpolate curr from SolutionVector to volume quadrature points and store in
to_interpV. Interpolate curr from SolutionVector to face quadrature points and
store in to_interpF.
"""
function interpolate!(solver::Solver, curr::SolutionVector, to_interpV, to_interpF)
    nstates = true_size(curr, 2)
    # First grab solution on faces and pack into array on_faces
    on_faces = Array{Float64,3}(undef, dofsF(solver),nstates,Geometry.N_FACES)
    for iK = 1:nelements(solver)
        for iF = 1:Geometry.N_FACES
            for iS = 1:nstates
                for iFN = 1:dofsF(solver)
                    on_faces[iFN,iS,iF] = curr.data[solver.mesh.ef2n[iFN,iF],iS,iK]
                end
            end
        end
        # Face interpolation: to_interpF = InterpF*on_faces
        on_face_vecs = reshape(on_faces, dofsF(solver), nstates*Geometry.N_FACES)
        @views to_interpF_vecs = reshape(to_interpF[:,:,:,iK], nQF(solver), nstates*Geometry.N_FACES)
        mul!(to_interpF_vecs, solver.InterpF,on_face_vecs)

        # Volume interpolation: to_interpV = InterpV*curr
        curr_vecs = @view curr.data[:,:,iK]
        to_interpV_vecs = @view to_interpV[:,:,iK]
        mul!(to_interpV_vecs, solver.InterpV,curr_vecs)
    end
end

"""
    convert_to_dim_array!(∇u, ∇u_compute)
Update ∇u (an array of `SolutionVector`s, one for each direction) from the
computed values stored in the single, long `SolutionVector` ∇u_compute
"""
function convert_to_dim_array!(∇u, ∇u_compute)
    (n, ns, ne) = true_size(∇u[1])
    dim = length(∇u)
    (n_times_dim, nsc, nec) = true_size(∇u_compute)
    @assert n*dim == n_times_dim && ns == nsc && ne == nec
    for iK = 1:ne
        for iS = 1:ns
            for iN = 1:n
                for l = 1:dim
                    ∇u[l].data[iN,iS,iK] = ∇u_compute.data[(iN-1)*dim+l,iS,iK]
                end
            end
        end
    end
end

"""
    local_flux!(solver::Solver)
Compute solver.∇u += F_l(u), the local DG flux for use in formulation of
second-order terms. Uses a downwind formulation for flux term.
"""
function local_flux!(solver::Solver)
    uK = zeros(nstates(solver))
    uN = zeros(nstates(solver))
    fstar = Array{Float64,2}(undef, nstates(solver),nQF(solver))
    JWI = Array{Float64,2}(undef, nQF(solver),dofsF(solver))
    flux_contribution = Array{Float64,2}(undef, nstates(solver),dofsF(solver))

    for iK = 1:nelements(solver)
        # There are l equations to handle for local flux term
        for l = 1:Geometry.DIM
            for iF = 1:Geometry.N_FACES
                nK = solver.mesh.e2e[iF,iK]
                normalK = solver.mesh.normals[l,iF,iK]

                if nK < 0 # Boundary condition
                    for iFQ = 1:nQF(solver)
                        uK .= solver.u_interpF[iFQ,:,iF,iK]
                        @views boundary_flux_l!(fstar[:,iFQ], solver.app, uK, nK, normalK)
                    end
                else # Inter-element flux_l
                    nF = solver.mesh.e2f[iF,iK]
                    for iFQ = 1:nQF(solver)
                        uK .= solver.u_interpF[iFQ                    ,:,iF,iK]
                        uN .= solver.u_interpF[solver.mesh.nfq2fq[iFQ],:,nF,nK]
                        @views numerical_flux_l!(fstar[:,iFQ], solver.app, uK, uN, solver.mesh.ldg_switch[iF,iK], normalK)
                    end
                end
                # JWI = Diag(J)*Diag(w)*InterpF
                mul!(JWI, Diagonal(solver.wQF .* solver.mesh.JkFdet[:,iF,iK]), solver.InterpF)
                # flux contribution = fstar*JWI
                mul!(flux_contribution, fstar, JWI)
                # Add up local face contributions into global ∇u array
                for iFN = 1:dofsF(solver)
                    for iS = 1:nstates(solver)
                        solver.∇u[l].data[solver.mesh.ef2n[iFN,iF],iS,iK] += flux_contribution[iS,iFN]
                    end
                end
            end
        end
    end
end

"""
    source_term!(du, solver::Solver, t)
Compute du = source(x,t) within element volumes
"""
function source_term!(du, solver::Solver, t)
    JWI = Array{Float64,2}(undef, nQV(solver),dofsV(solver))
    f = Array{Float64,2}(undef, nstates(solver),nQV(solver))
    for iK = 1:nelements(solver)
        # JWI = Diag(J)*Diag(w)*InterpV
        mul!(JWI, Diagonal(solver.wQV .* solver.mesh.Jkdet[:,iK]), solver.InterpV)

        for iQ = 1:nQV(solver)
            @views source!(f[:,iQ], solver.app, t)
        end
        # du[:,:,iK] = JWI'*f
        @views mul!(du.data[:,:,iK], JWI', f')
    end
end

"""
    convect_flux!(du, solver::Solver)
Compute du -= Fc(u), which is the 1st order (convection) flux term along faces
"""
function convect_flux!(du, solver::Solver)
    uK = zeros(nstates(solver))
    uN = zeros(nstates(solver))
    fstar = Array{Float64,2}(undef, nstates(solver),nQF(solver))
    JWI = Array{Float64,2}(undef, nQF(solver),dofsF(solver))
    flux_contribution = Array{Float64,2}(undef, nstates(solver),dofsF(solver))

    for iK = 1:nelements(solver)
        for iF = 1:Geometry.N_FACES
            nK = solver.mesh.e2e[iF,iK]
            normalK = @view solver.mesh.normals[:,iF,iK]

            if nK < 0 # Boundary condition
                for iFQ = 1:nQF(solver)
                    uK .= solver.u_interpF[iFQ,:,iF,iK]
                    @views boundary_flux_c!(fstar[:,iFQ], solver.app, uK, nK, normalK)
                end
            else # Inter-element flux_c
                nF = solver.mesh.e2f[iF,iK]
                for iFQ = 1:nQF(solver)
                    uK .= solver.u_interpF[iFQ                    ,:,iF,iK]
                    uN .= solver.u_interpF[solver.mesh.nfq2fq[iFQ],:,nF,nK]
                    @views numerical_flux_c!(fstar[:,iFQ], solver.app, uK, uN, normalK)
                end
            end

            # JWI = Diag(J)*Diag(w)*InterpF
            mul!(JWI, Diagonal(solver.wQF .* solver.mesh.JkFdet[:,iF,iK]), solver.InterpF)
            # flux contribution = fstar*JWI
            mul!(flux_contribution, fstar, JWI)
            # Add up local face contributions into global du array
            for iFN = 1:dofsF(solver)
                for iS = 1:nstates(solver)
                    du.data[solver.mesh.ef2n[iFN,iF],iS,iK] -= flux_contribution[iS,iFN]
                end
            end
        end
    end
end

"""
    convect_volume!(du, solver::Solver)
Compute du += Kc(u), which is the 1st order (convection) local volume integral
within elements
"""
function convect_volume!(du, solver::Solver)
    uK = zeros(nstates(solver))
    fc = Array{Float64,3}(undef, nstates(solver), Geometry.DIM, nQV(solver))
    grad = Array{Float64,3}(undef, Geometry.DIM, nstates(solver), nQV(solver))
    # If not using Julia *, should initialize with DIM in slowest dimension

    for iK = 1:nelements(solver)
        for iQ = 1:nQV(solver)
            uK .= solver.u_interpV[iQ,:,iK]
            @views flux_c!(fc[:,:,iQ], solver.app, uK)
        end

        # TODO: Eventually this should be done in matrix-free approach
        Jks = @view solver.mesh.Jk[:,:,:,iK]
        # op_q = wQ * J^{-1}*det(J) == wQ * adj(J)
        op_q = similar(Jks)
        op_q[1,1,:] = solver.wQV.* Jks[2,2,:]
        op_q[1,2,:] = solver.wQV.*-Jks[1,2,:]
        op_q[2,1,:] = solver.wQV.*-Jks[2,1,:]
        op_q[2,2,:] = solver.wQV.* Jks[1,1,:]
        @views for iQ = 1:nQV(solver)
            # grad = op * fc'
            mul!(grad[:,:,iQ], op_q[:,:,iQ], fc[:,:,iQ]')
        end
        # du += dInterpV'*grad'
        @views mul!(du.data[:,:,iK], solver.dInterpV[:,:,1]', grad[1,:,:]', 1.0, 1.0)
        @views mul!(du.data[:,:,iK], solver.dInterpV[:,:,2]', grad[2,:,:]', 1.0, 1.0)
    end
end

"""
    function diffusive_flux!(du, solver)
Compute du += Fd(u,∇u), which is the 2nd order (diffusive) flux term along faces
"""
function diffusive_flux!(du, solver)
    uK = zeros(nstates(solver))
    uN = zeros(nstates(solver))
    DuK = zeros(nstates(solver), Geometry.DIM)
    DuN = zeros(nstates(solver), Geometry.DIM)
    fstar = Array{Float64,2}(undef, nstates(solver),nQF(solver))
    JWI = Array{Float64,2}(undef, nQF(solver),dofsF(solver))
    flux_contribution = Array{Float64,2}(undef, nstates(solver),dofsF(solver))

    for iK = 1:nelements(solver)
        for iF = 1:Geometry.N_FACES
            nK = solver.mesh.e2e[iF,iK]
            normalK = @view solver.mesh.normals[:,iF,iK]

            if nK < 0 # Boundary condition
                for iFQ = 1:nQF(solver)
                    uK .= solver.u_interpF[iFQ,:,iF,iK]
                    DuK .= solver.∇u_interpF[iFQ,:,iF,iK,:]
                    @views boundary_flux_d!(fstar[:,iFQ], solver.app, uK, DuK, nK, normalK)
                end
            else # Inter-element flux_c
                nF = solver.mesh.e2f[iF,iK]
                for iFQ = 1:nQF(solver)
                    uK .= solver.u_interpF[iFQ                    ,:,iF,iK]
                    uN .= solver.u_interpF[solver.mesh.nfq2fq[iFQ],:,nF,nK]
                    DuK .= solver.∇u_interpF[iFQ                    ,:,iF,iK,:]
                    DuN .= solver.∇u_interpF[solver.mesh.nfq2fq[iFQ],:,nF,nK,:]
                    @views numerical_flux_d!(fstar[:,iFQ], solver.app, uK, uN, DuK, DuN, solver.mesh.ldg_switch[iF,iK],normalK)
                end
            end

            # JWI = Diag(J)*Diag(w)*InterpF
            mul!(JWI, Diagonal(solver.wQF .* solver.mesh.JkFdet[:,iF,iK]), solver.InterpF)
            # flux contribution = fstar*JWI
            mul!(flux_contribution, fstar, JWI)
            # Add up local face contributions into global du array
            for iFN = 1:dofsF(solver)
                for iS = 1:nstates(solver)
                    du.data[solver.mesh.ef2n[iFN,iF],iS,iK] += flux_contribution[iS,iFN]
                end
            end
        end
    end
end

"""
    diffusive_volume!(du, solver::Solver)
Compute du -= Kd(u,∇u), which is the 2nd order (diffusive) local volume integral
within elements
"""
function diffusive_volume!(du, solver::Solver)
    uK = zeros(nstates(solver))
    DuK = zeros(nstates(solver),Geometry.DIM)
    fd = Array{Float64,3}(undef, nstates(solver), Geometry.DIM, nQV(solver))
    grad = Array{Float64,3}(undef, Geometry.DIM, nstates(solver), nQV(solver))
    # If not using Julia *, should initialize with DIM in slowest dimension

    for iK = 1:nelements(solver)
        for iQ = 1:nQV(solver)
            uK .= solver.u_interpV[iQ,:,iK]
            DuK .= solver.∇u_interpV[iQ,:,iK,:]
            @views flux_d!(fd[:,:,iQ], solver.app, uK, DuK)
        end

        # TODO: Eventually this should be done in matrix-free approach
        Jks = @view solver.mesh.Jk[:,:,:,iK]
        # op_q = wQ * J^{-1}*det(J) == wQ * adj(J)
        op_q = similar(Jks)
        op_q[1,1,:] = solver.wQV.* Jks[2,2,:]
        op_q[1,2,:] = solver.wQV.*-Jks[1,2,:]
        op_q[2,1,:] = solver.wQV.*-Jks[2,1,:]
        op_q[2,2,:] = solver.wQV.* Jks[1,1,:]
        @views for iQ = 1:nQV(solver)
            # grad = op * fd'
            mul!(grad[:,:,iQ], op_q[:,:,iQ], fd[:,:,iQ]')
        end
        # du -= dInterpV'*grad'
        @views mul!(du.data[:,:,iK], solver.dInterpV[:,:,1]', grad[1,:,:]', -1.0, 1.0)
        @views mul!(du.data[:,:,iK], solver.dInterpV[:,:,2]', grad[2,:,:]', -1.0, 1.0)
    end
end

include("FEImplicit.jl")

end # module FiniteElement
