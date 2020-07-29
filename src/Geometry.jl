"""
    Module Geometry
All mesh information and setup related to element connectivity
"""
module Geometry

import LinearAlgebra.norm

export Mesh
export periodic_cube

const DIM = 2
const N_FACES = 2*DIM
const N_VERTICES = 2^DIM
const N_FVERTICES = 2^(DIM-1)

"""
    struct Mesh
Contains all mesh information related to domain and element connectivity
"""
mutable struct Mesh
    # Global number of elements in mesh
    n_elements::Int
    # Global number of vertices in mesh
    n_vertices::Int
    # Globally minimum edge length across all elements
    min_dx::Float64
    # Global vector of N-D vertices that form corners of elements
    vertices
    # Element-to-Vertex mapping:
    # For every element iK, vertex i, e2v[i,iK] = vertex index of local vertex i
    e2v
    # Face-to-Vertex mapping:
    # For every face iF, face vertex i, f2v[i,iF] = this element's vertex index
    f2v
    # Element-to-Element neighbor mapping:
    # For every element iK, face i, e2e[i,iK] = element index of neighbor on ith face
    e2e
    # Element-to-Face mapping:
    # For every element iK, face i, e2f[i,iK] = face index on neighbor's face array
    e2f
    # Normals mapping:
    # For every element iK, face i, normals[:,i,iK] = outward normal vector of face i
    normals
    # Bilinear mapping:
    # For every element iK, dimension l
    # x_l = sum(bilinear_mapping[:,l,iK]*phi[xi_l,:])
    bilinear_mapping

    ######################################
    # Initialized after solver is created
    ######################################
    # Order of DG method
    order::Int
    # Local number of DG nodes per element
    n_nodes::Int
    # Local number of DG nodes per element face
    n_face_nodes::Int
    # Local number of quadrature points per element face
    n_face_quad_nodes::Int
    # True coordinates of DG nodes for every element:
    # For every element iK, node i, global_coords[:,i,iK] = ND coordinate of node i
    global_coords
    # Element face to node mapping:
    # For every face i (same mapping across all elements), ef2n[:,i] = local
    # node IDs of nodes on face i
    ef2n
    # Neighbor face node to face node mapping
    # For every face node iFN (same mapping across all faces/elements),
    # nfn2fn[iFN] = face node ID on neighbor's face
    # Neighbor node: solution[nfn2fn[iFN], e2f[iF,iK], :, e2e[iF,iK]]
    nfn2fn
    # Element face to quadrature point mapping:
    # For every face i (same mapping across all elements), ef2q[:,i] = local
    # quad IDs of quadrature points on face i
    ef2q
    # Neighbor face quadrature point to quadrature point mapping:
    # For every face quadrature point ID iFQ (same mapping across all faces/elements),
    # nfq2fq[iFQ] = face quadrature point ID on neighbor's face
    # Neighbor quad: solution[nfq2fq[iFQ], e2f[iF,iK], :, e2e[iF,iK]]
    nfq2fq

    function Mesh(ne_, nv_, mdx_, v_, e2v_, f2v_, e2e_, e2f_, normals_, bm_)
        return new(ne_, nv_, mdx_, v_, e2v_, f2v_, e2e_, e2f_, normals_, bm_,
            0,0,0,0,[],[],[],[],[])
    end
end

"""
    periodic_square(ns::Array)
Creates a periodic square [0,1]^2 with prod(ns) global number of elements
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
Labels faces of a rectangle according to CCW vertices
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
Computes the bilinear mapping from element coordinates
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
Returns the normals from the element coordinates, assuming 2D mesh is ordered CCW.
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

end
