"""
    Module Geometry
All mesh information and setup related to element connectivity
"""
module Geometry

import LinearAlgebra.norm
import Basis1D
import Basis2D.dPhi2D

export Mesh
export periodic_square
export setup_nodes!, setup_quads!
export precompute_jacobians!, precompute_ldg_switch!
export output_mesh

" Dimension of overall mesh "
const DIM = 2
" Number of faces per element "
const N_FACES = 2*DIM
" Number of vertices per element "
const N_VERTICES = 2^DIM
" Number of vertices per element face "
const N_FVERTICES = 2^(DIM-1)

"""
    struct Mesh
Contains all mesh information related to domain and element connectivity
"""
mutable struct Mesh
    " Global number of elements in mesh "
    n_elements::Int
    " Global number of vertices in mesh "
    n_vertices::Int
    " Globally minimum edge length across all elements "
    min_dx::Float64
    " Global vector of N-D vertices that form corners of elements "
    vertices
    """ Element-to-Vertex mapping:
    For every element iK, vertex i, `e2v[i,iK]` = vertex index of local vertex i """
    e2v
    """ Face-to-Vertex mapping:
    For every face iF, face vertex i, `f2v[i,iF]` = this element's vertex index """
    f2v
    """ Element-to-Element neighbor mapping:
    For every element iK, face iF, `e2e[iF,iK]` = element index of neighbor on face iF """
    e2e
    """ Element-to-Face mapping:
    For every element iK, face iF, `e2f[iF,iK]` = face index on neighbor's face array """
    e2f
    """ Normals mapping:
    For every element iK, face iF, `normals[:,iF,iK]` = outward normal vector of face iF """
    normals
    """
    Bilinear mapping:

    For every element iK, dimension l,
    ``x_l`` = `sum(bilinear_mapping[:,l,iK]*phi[xi_l,:])`
    """
    bilinear_mapping

    ######################################
    # Initialized after solver is created
    ######################################
    " Order of DG method "
    order::Int
    " Local number of DG nodes per element "
    n_nodes::Int
    " Local number of quadrature points per element "
    n_quads::Int
    " Local number of DG nodes per element face "
    n_face_nodes::Int
    " Local number of quadrature points per element face "
    n_face_quads::Int
    """
    True coordinates of DG nodes for every element

    For every element iK, node i, `coord_nodes[:,i,iK]` = ND coordinate of node i
    """
    coord_nodes
    """
    True coordinates of DG quadrature points for every element

    For every element iK, quad point i, `coord_quads[:,i,iK]` = ND
    coordinate of quad point i
    """
    coord_quads
    """
    Jacobian of transformation evaluated at each quadrature point, element

    For every element iK, quad point iQ, `Jk[:,:,iQ,iK]` ∈ ℜ^(DIM x DIM)
    is the Jacobian of the mapping from the reference element to element iK at
    that quadrature point.
    """
    Jk
    """
    |det Jacobian| of transformation evaluated each quadrature point, element

    For every element iK, quad point iQ, `Jkdet[iQ,iK]` is |det Jac| of the
    mapping from the reference element to element iK at quad point iQ.
    """
    Jkdet
    """
    Jacobian of transformation evaluated at each face quadrature point, element

    For every element iK, face iF, face quad point iQ, `JkF[:,:,iQ,iF,iK]`
    ∈ ℜ^(DIM x DIM) is the Jacobian of the mapping from the reference face
    element to face iF of element iK at face quad point iQ
    """
    JkF
    """
    |det Jacobian| of face integral evaluated at each face quadrature point, element

    For every element iK, face iF, face quad point iQ, `JkFdet[iQ,iF,iK]` is
    |gamma'(t)| from reference face element to face iF at face quad point iQ.
    """
    JkFdet
    """
    Element face to node mapping

    For every face iF (same mapping across all elements), face node iFN,
    `ef2n[iFN,iF]` = local node ID of face node iFN on face iF
    """
    ef2n
    """
    Neighbor face node to face node mapping

    For every face node iFN (same mapping across all faces/elements),
    `nfn2fn[iFN]` = face node ID on neighbor's face

    Neighbor node: `solution[nfn2fn[iFN], e2f[iF,iK], :, e2e[iF,iK]]`
    """
    nfn2fn
    """
    Element face to quadrature point mapping

    For every face iF (same mapping across all elements), face quad iFQ,
    `ef2q[iFQ,iF]` = local quad ID of face quad iFQ on face iF
    """
    ef2q
    """
    Neighbor face quadrature point to quadrature point mapping

    For every face quadrature point ID iFQ (same mapping across all faces/elements),
    `nfq2fq[iFQ]` = face quadrature point ID on neighbor's face

    Neighbor quad: `solution[nfq2fq[iFQ], e2f[iF,iK], :, e2e[iF,iK]]`
    """
    nfq2fq
    """
    Local DG switch (only needed for 2nd-order applications):

    For every element iK, face iF, `ldg_switch[iF,iK]` = true if ldg flux
    should be taken from this element's face values
    """
    ldg_switch

    function Mesh(ne_, nv_, mdx_, v_, e2v_, f2v_, e2e_, e2f_, normals_, bm_)
        return new(ne_, nv_, mdx_, v_, e2v_, f2v_, e2e_, e2f_, normals_, bm_,
            0,0,0,0,0,[],[],[],[],[],[],[],[],[],[],[])
    end
end

"""
    Base.show(io::IO, mesh::Mesh)
Allow for pretty printing of mesh type
"""
function Base.show(io::IO, mesh::Mesh)
    print(io, "$(mesh.n_vertices) vertices connected with $(mesh.n_elements) elements.")
end

"""
    periodic_square(ns::Array)
Create a periodic square [0,1]^2 with prod(ns) global number of elements
"""
function periodic_square(ns)::Mesh
    @assert length(ns) == DIM
    top_right = ones(DIM)
    bot_left  = zeros(DIM)
    n_elements = prod(ns)
    n_vertices = prod(ns.+1)
    min_edge_length = minimum((top_right .- bot_left)./ns)

    # Initialize vertices
    nd_vertices = Array{Float64,DIM+1}(undef,DIM,(ns.+1)...)
    for iy = 1:ns[2]+1
        curry = (iy-1)*(top_right[2]-bot_left[2])/ns[2]+bot_left[2]
        for ix = 1:ns[1]+1
            currx = (ix-1)*(top_right[1]-bot_left[1])/ns[1]+bot_left[1]
            nd_vertices[1,ix,iy] = currx
            nd_vertices[2,ix,iy] = curry
        end
    end
    vertices = reshape(nd_vertices, DIM, n_vertices)

    # Initialize element to vertices array
    e2v = Array{Int,2}(undef, N_VERTICES, n_elements)
    for iy = 1:ns[2]
        yoff1 = (iy-1)*(ns[1]+1)
        yoff2 = (iy  )*(ns[1]+1)
        for ix = 1:ns[1]
            xoff1 = ix # Only fastest direction contains the 1-indexing
            xoff2 = ix+1
            e_index = ix+(iy-1)*ns[1]
            # Stores vertices in CCW order
            e2v[1, e_index] = xoff1+yoff2
            e2v[2, e_index] = xoff1+yoff1
            e2v[3, e_index] = xoff2+yoff1
            e2v[4, e_index] = xoff2+yoff2
        end
    end

    # Initialize element to face arrays
    e2e = Array{Int, 2}(undef, N_FACES, n_elements)
    e2f = Array{Int, 2}(undef, N_FACES, n_elements)
    for iy = 1:ns[2]
        iyM = mod(iy-2,ns[2]) * ns[1]
        iyP = mod(iy,  ns[2]) * ns[1]
        iy0 = (iy-1) * ns[1]
        for ix = 1:ns[1]
            # Only fastest direction contains the 1-indexing
            ixM = mod(ix-2,ns[1])+1
            ixP = mod(ix,  ns[1])+1
            ix0 = ix
            e_index = ix0+iy0

            # Neighbor elements are stored in -x,-y,+x,+y order to be CCW
            e2e[1,e_index] = ixM+iy0
            e2e[2,e_index] = ix0+iyM
            e2e[3,e_index] = ixP+iy0
            e2e[4,e_index] = ix0+iyP

            # Face ID of this element's -x face will be neighbor's +x face
            e2f[1,e_index] = 3
            e2f[2,e_index] = 4
            e2f[3,e_index] = 1
            e2f[4,e_index] = 2
        end
    end

    # Initialize f2v
    f2v = init_f2v()

    # Initialize bilinear_mapping
    bilinear_mapping = init_bilinear_mapping(vertices,e2v)

    # Initialize normals and min_dx
    (normals, min_dx) = init_normals(vertices,f2v,e2v)

    mesh = Mesh(n_elements, n_vertices, min_dx, vertices, e2v, f2v, e2e, e2f, normals, bilinear_mapping)
    return mesh
end

"""
    init_f2v()
Label faces of a rectangle according to CCW vertices
"""
function init_f2v()
    f2v = Array{Int,2}(undef, N_FVERTICES, N_FACES)
    # Mapping assumes input vertices are CCW
    for iF = 1:N_FACES
        f2v[1,iF] = iF
        f2v[2,iF] = mod(iF, N_FACES)+1
    end
    return f2v
end

"""
    init_bilinear_mapping(vertices,e2v)
Compute the bilinear mapping from element coordinates
"""
function init_bilinear_mapping(vertices, e2v)
    map_order = 1
    map_dofs = (map_order+1)^DIM
    # Sanity check that bilinear mapping is the correct size
    @assert map_dofs == N_VERTICES

    # Gmsh =   4---3   Mapping =  3---4
    # CCW      |   |   wants      |   |
    # corners  1---2              1---2
    c2m = [1,2,4,3]
    bilinear_mapping = Array{Float64,3}(undef, N_VERTICES, DIM, size(e2v,2))
    for iK = 1:size(e2v,2)
        for l = 1:DIM
            for iV = 1:N_VERTICES
                bilinear_mapping[c2m[iV],l,iK] = vertices[l,e2v[iV,iK]]
            end
        end
    end
    return bilinear_mapping
end

"""
    init_normals(vertices, f2v, e2v)
Return the normals from the element coordinates, assuming 2D mesh is ordered CCW.
Simultaneously computes the minimum edge length across global mesh.
"""
function init_normals(vertices, f2v, e2v)
    n_elements = size(e2v,2)
    normals = zeros(Float64, DIM, N_FACES, n_elements)
    min_dx = typemax(Float64)
    for iK = 1:n_elements
        for iF = 1:N_FACES
            a = f2v[1,iF]
            b = f2v[2,iF]
            AB = vertices[:,e2v[b,iK]].-vertices[:,e2v[a,iK]]
            len = norm(AB,2)
            # Normal to (dx,dy): (dy,-dx)
            normals[1,iF,iK] = AB[2]/len
            normals[2,iF,iK] = -AB[1]/len

            min_dx = min(min_dx, len) # Compute minimum edge length
        end
    end
    return (normals, min_dx)
end

"""
    setup_nodes!(mesh::Mesh, InterpTk, order; compute_global=false)
Initialize element nodes within Mesh. If `compute_global`, then also compute
the global nodes from bilinear mapping of reference nodes and store in
`coord_nodes`. This would be needed if solutions, for instance, to be stored at
nodes would be needed to view.
"""
function setup_nodes!(mesh::Mesh, InterpTk, order; compute_global=false)
    mesh.order = order
    mesh.n_nodes = size(InterpTk,1)
    if compute_global
        mesh.coord_nodes = Array{Float64,3}(undef, DIM, mesh.n_nodes, mesh.n_elements)
        for iK = 1:mesh.n_elements
            for l = 1:DIM
                mesh.coord_nodes[l,:,iK] = InterpTk*mesh.bilinear_mapping[:,l,iK]
            end
        end
    end

    (mesh.n_face_nodes,mesh.ef2n,mesh.nfn2fn) = init_face_map(mesh, mesh.order+1)
end

"""
    setup_quads!(mesh::Mesh, InterpTkQ; compute_global=false)
Initialize element quadrature points within Mesh. If `compute_global`, then also
compute global quadrature points from bilinear mapping of reference quad points
and store in `coord_quads`. This would be needed if material properties, for instance,
to be stored at quadrature points would be needed.
"""
function setup_quads!(mesh::Mesh, InterpTkQ; compute_global=false)
    if mesh.order <= 0
        error("Mesh must setup nodes before quads!")
    end
    mesh.n_quads = size(InterpTkQ,1)
    if compute_global
        mesh.coord_quads = Array{Float64,3}(undef, DIM, mesh.n_quads, mesh.n_elements)
        for iK = 1:mesh.n_elements
            for l = 1:DIM
                mesh.coord_quads[l,:,iK] = InterpTkQ*mesh.bilinear_mapping[:,l,iK]
            end
        end
    end

    nQ = ceil(Int, mesh.order+1/2) # Quadrature points per face
    (mesh.n_face_quads, mesh.ef2q, mesh.nfq2fq) = init_face_map(mesh, nQ)
end

"""
    init_face_map(mesh::Mesh, size1D)
Initialize face maps for a given number of nodes.
Assumes that each face is a side with exactly size1D nodes.
For every element iK, face iF, face node iFN:

my node       @: `soln[efmap[iFN, iF], :, iK]`

neighbor node @: `soln[efmap[nfmap[iFN], e2f[iF,iK]], :, e2e[iF,iK]]`
"""
function init_face_map(mesh::Mesh, size1D)
    efmap = Array{Int64,2}(undef, size1D, N_FACES)
    # efmap has explicit dependence on f2v. Only handles CCW vertices
    # 4---3
    # |   |
    # 1---2
    for iF = 1:N_FACES
        if mesh.f2v[:,iF] == [4,1]
            # -x direction face
            x_off = 1 # 1-indexed
            y_off = (size1D-1:-1:0).*size1D
            efmap[:,iF] = x_off.+y_off
        elseif mesh.f2v[:,iF] == [1,2]
            # -y direction face
            y_off = 0
            x_off = 1:size1D # 1-indexed
            efmap[:,iF] = x_off.+y_off
        elseif mesh.f2v[:,iF] == [2,3]
            # +x direction face
            x_off = size1D # 1-indexed
            y_off = (0:size1D-1).*size1D
            efmap[:,iF] = x_off.+y_off
        elseif mesh.f2v[:,iF] == [3,4]
            # +y direction face
            y_off = (size1D-1)*size1D
            x_off = size1D:-1:1 # 1-indexed
            efmap[:,iF] = x_off.+y_off
        else
            error("Vertices for f2v are not CCW!")
        end
    end

    # nfmap is reversed version of efmap
    nfmap = collect(size1D:-1:1)

    # Face of 2D mesh is 1D
    fnodes = size1D
    return (fnodes,efmap,nfmap)
end

"""
    precompute_jacobians!(mesh::Mesh, xTk1D, xQ1D)
Precompute Jacobian of mappings from reference element to all elements,
evaluated at quadrature points.

Depending on the specified flags, will
initialize mesh.Jk (Jacobians), mesh.Jkdet (|det Jac|), mesh.JkF (face Jacobians),
and/or mesh.JkFdet(|det face Jac|).

xTk1D is 1D points that bilinear_mapping Tk is defined on
xQ1D is 1D quadrature points, and assumes faces are xQ1D, while volume is (xQ1D,xQ1D)
"""
function precompute_jacobians!(mesh::Mesh, xTk1D, xQ1D)
    # Compute grad of mapping from reference bases, evaluated on element quad points
    xyTk = (xTk1D,xTk1D)
    dPhiTkQ = dPhi2D(xyTk, (xQ1D,xQ1D))

    # Compute grad of mapping from reference bases, evaluated on face quad points
    n_from = prod(size.(xyTk,1))
    n_to = size(xQ1D,1) # 1D face
    dPhiTkQF = Array{Float64,4}(undef, n_to,n_from,DIM,N_FACES)
    for iF = 1:N_FACES
        # xyQF = "kronecker product" that represents face quad points on face iF
        if mesh.f2v[:,iF] == [4,1]
            # -x direction face
            xQF = [xTk1D[1]]
            yQF = reverse(xQ1D)
        elseif mesh.f2v[:,iF] == [1,2]
            # -y direction face
            xQF = xQ1D
            yQF = [xTk1D[1]]
        elseif mesh.f2v[:,iF] == [2,3]
            # +x direction face
            xQF = [xTk1D[2]]
            yQF = xQ1D
        elseif mesh.f2v[:,iF] == [3,4]
            # +y direction face
            xQF = reverse(xQ1D)
            yQF = [xTk1D[2]]
        else
            error("Vertices for f2v are not CCW!")
        end
        xyQF = (xQF,yQF)
        dPhiTkQF[:,:,:,iF] = dPhi2D(xyTk, xyQF)
    end

    # Jacobian = dPhiTkQ[:,:,lj]*bilinear_mapping[:,li,iK]
    mesh.Jk = Array{Float64,4}(undef, DIM,DIM,mesh.n_quads,mesh.n_elements)
    mesh.Jkdet = Array{Float64,2}(undef, mesh.n_quads,mesh.n_elements)
    for iK = 1:mesh.n_elements
        for li = 1:DIM
            for lj = 1:DIM
                mesh.Jk[li,lj,:,iK] = dPhiTkQ[:,:,lj]*mesh.bilinear_mapping[:,li,iK]
            end
        end
        Jac = @view mesh.Jk[:,:,:,iK]
        mesh.Jkdet[:,iK] = Jac[1,1,:].*Jac[2,2,:].-Jac[2,1,:].*Jac[1,2,:]
    end
    # Face Jacobian = dPhiTkQF[:,:,lj,iF]*bilinear_mapping[:,li,iK]
    mesh.JkF = Array{Float64,5}(undef, DIM,DIM,mesh.n_face_quads,N_FACES,mesh.n_elements)
    mesh.JkFdet = Array{Float64,3}(undef, mesh.n_face_quads,N_FACES,mesh.n_elements)
    for iK = 1:mesh.n_elements
        for iF = 1:N_FACES
            for li = 1:DIM
                for lj = 1:DIM
                    mesh.JkF[li,lj,:,iF,iK] = dPhiTkQF[:,:,lj,iF]*mesh.bilinear_mapping[:,li,iK]
                end
            end
            # 1D integrals => only needs |gamma'(t)| term along correct x_i_{l_i}
            lj = 0
            if mesh.f2v[:,iF] == [4,1]
                # -x direction face
                lj = 2
            elseif mesh.f2v[:,iF] == [1,2]
                # -y direction face
                lj = 1
            elseif mesh.f2v[:,iF] == [2,3]
                # +x direction face
                lj = 2
            elseif mesh.f2v[:,iF] == [3,4]
                # +y direction face
                lj = 1
            else
                error("Vertices for f2v are not CCW!")
            end
            for iQ = 1:mesh.n_face_quads
                mesh.JkFdet[iQ,iF,iK] = norm(mesh.JkF[:,lj,iQ,iF,iK],2)
            end
        end
    end
end

"""
    function precompute_ldg_switch!(mesh::Mesh)
Precompute the Local DG switch for 2nd-order applications. Opposite faces of
each element are guaranteed to have opposing switch values.
"""
function precompute_ldg_switch!(mesh::Mesh)
    # Initialize opposite_face array
    opposite_face = zeros(Int64,Geometry.N_FACES)
    for iF = 1:Geometry.N_FACES
        for iF2 = 1:Geometry.N_FACES
            # Opposite faces share no common vertices
            if isempty(intersect(mesh.f2v[:,iF],mesh.f2v[:,iF2]))
                opposite_face[iF] = iF2
                break
            end
        end
    end

    visited = falses(Geometry.N_FACES, mesh.n_elements)
    mesh.ldg_switch = zeros(Bool, Geometry.N_FACES, mesh.n_elements)
    while true
        index = findfirst(x -> !x, visited)
        if isnothing(index)
            break
        end
        iF = index[1]; iK = index[2];
        init_iF = iF; init_iK = iK
        choose_this_element = true
        while true
            mesh.ldg_switch[iF,iK] = choose_this_element
            visited[iF,iK] = true
            nK = mesh.e2e[iF,iK]
            if nK < 0 # Boundary
                if choose_this_element # Now reverse direction from initial choice
                    choose_this_element = !choose_this_element
                    iK = init_iK
                    iF = opposite_face[init_iF]
                else # We have already reversed once
                    break
                end
            else # Inter-element border
                nF = mesh.e2f[iF,iK]
                mesh.ldg_switch[nF,nK] = !choose_this_element
                visited[nF,nK] = true
                iK = nK
                iF = opposite_face[nF]
                if visited[iF,iK]
                    if choose_this_element # Now reverse direction from initial choice
                        choose_this_element = !choose_this_element
                        iK = init_iK
                        iF = opposite_face[init_iF]
                    else # We have already reversed once
                        break
                    end
                end
            end
        end
    end
end

"""
    output_mesh(mesh::Mesh, filename)
Outputs the given mesh as a Gmsh 4.1 file format .msh file
"""
function output_mesh(mesh::Mesh, filename)
    file = open(filename, "w")
    # Uses 4.1 MSH file format from https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
    println(file, "\$MeshFormat")
    println(file, "4.1 0 8") # version 4.1, ASCII
    println(file, "\$EndMeshFormat")

    # Vertices (mesh.vertices)
    println(file, "\$Nodes")
    #     num_entities num_vertices start_vertex end_vertex
    println(file, "1 $(mesh.n_vertices) 1 $(mesh.n_vertices)")
    #     2D entity (surface) #1, no parametric coords, num_vertices in entity
    println(file, "2 1 0 $(mesh.n_vertices)")
    for iV = 1:mesh.n_vertices
        println(file, "$iV")
    end
    for iV = 1:mesh.n_vertices
        # Vertices must be in 3D
        println(file, "$(mesh.vertices[1,iV]) $(mesh.vertices[2,iV]) 0.0")
    end
    println(file, "\$EndNodes")

    # Elements (mesh.e2v)
    println(file, "\$Elements")
    #     num_entities num_elements start_element end_element
    println(file, "1 $(mesh.n_elements) 1 $(mesh.n_elements)")
    #     2D entity (surface) #1, element type 3 (4-coord quad), num_elements
    println(file, "2 1 3 $(mesh.n_elements)")
    for iK = 1:size(mesh.e2v,2)
        print(file, "$iK")
        for iV = 1:size(mesh.e2v,1)
            print(file, " $(mesh.e2v[iV,iK])")
        end
        println(file)
    end
    println(file, "\$EndElements")
    # TODO: add periodicity if applicable

    close(file)
end

end
