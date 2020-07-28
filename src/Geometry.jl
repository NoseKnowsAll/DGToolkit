"""
    Module Geometry
All mesh information and setup related to element connectivity
"""
module Geometry

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
    min_edge_length::Float64
    # Global vector of N-D vertices that form corners of elements
    vertices
    # Element-to-Vertex mapping:
    # For every element k, vertex i, e2v[i,k] = vertex index of local vertex i
    e2v
    # Face-to-Vertex mapping:
    # For every face j, face vertex i, f2v[i,f] = element vertex index
    f2v
    # Element-to-Element neighbor mapping:
    # For every element k, face i, e2e[i,k] = element index of neighbor on ith face
    e2e
    # Element-to-Face mapping:
    # For every element k, face i, e2f[i,k] = face index on neighbor's face array
    e2f
    # Normals mapping:
    # For every element k, face i, normals[:,i,k] = outward normal vector of face i
    normals
    # Bilinear mapping:
    # For every element k, dimension l
    # x_l = sum(bilinear_mapping[:,l,k]*phi[xi_l,:])
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
    # For every element k, node i, global_coords[:,i,k] = ND location of node i
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

    Mesh() = new(0,0,0,[],[],[],[],[],0,0,0,0,[],[],[])
    function Mesh(ne_, nv_, mel_, v_, e2v_, e2e_, e2f_, normals_)
        return new(ne_, nv_, mel_, v_, e2v_, e2e_, e2f_, normals_,0,0,0,0,[],[],[])
    end
end

# Creates a periodic cube [0,1]^d with prod(ns) global number of elements
function periodic_cube(ns)::Mesh
    @assert length(ns) == DIM
    top_right = ones(DIM)
    bot_left  = zeros(DIM)
    n_elements = prod(ns)
    n_vertices = prod(ns.+1)
    min_edge_length = minimum((top_right .- bot_left)./ns)

    # Initialize vertices
    nd_vertices = Array{Float64,DIM+1}(undef,DIM,(ns.+1)...)
    for iz = 1:ns[3]+1
        currz = (iz-1)*(top_right[3]-bot_left[3])/ns[3]+bot_left[3]
        for iy = 1:ns[2]+1
            curry = (iy-1)*(top_right[2]-bot_left[2])/ns[2]+bot_left[2]
            for ix = 1:ns[1]+1
                currx = (ix-1)*(top_right[1]-bot_left[1])/ns[1]+bot_left[1]
                nd_vertices[1,ix,iy,iz] = currx
                nd_vertices[2,ix,iy,iz] = curry
                nd_vertices[3,ix,iy,iz] = currz
            end
        end
    end
    vertices = reshape(nd_vertices, DIM, n_vertices)

    # Initialize element to vertices array
    e2v = Array{Int,2}(undef, N_VERTICES, n_elements)
    for iz = 1:ns[3]
        zoff1 = (iz-1)*(ns[1]+1)*(ns[2]+1)
        zoff2 = (iz  )*(ns[1]+1)*(ns[2]+1)
        for iy = 1:ns[2]
            yoff1 = (iy-1)*(ns[1]+1)
            yoff2 = (iy  )*(ns[1]+1)
            for ix = 1:ns[1]
                xoff1 = ix # Only fastest direction contains the 1-indexing
                xoff2 = ix+1
                e_index = ix+(iy-1)*ns[1]+(iz-1)*ns[1]*ns[2]
                e2v[1, e_index] = xoff1+yoff1+zoff1
                e2v[2, e_index] = xoff2+yoff1+zoff1
                e2v[3, e_index] = xoff1+yoff2+zoff1
                e2v[4, e_index] = xoff2+yoff2+zoff1
                e2v[5, e_index] = xoff1+yoff1+zoff2
                e2v[6, e_index] = xoff2+yoff1+zoff2
                e2v[7, e_index] = xoff1+yoff2+zoff2
                e2v[8, e_index] = xoff2+yoff2+zoff2
            end
        end
    end

    # Initialize element to face arrays
    e2e = Array{Int, 2}(undef, N_FACES, n_elements)
    e2f = Array{Int, 2}(undef, N_FACES, n_elements)
    for iz = 1:ns[3]
        izM = mod(iz-2,ns[3]) * (ns[1]*ns[2])
        izP = mod(iz,  ns[3]) * (ns[1]*ns[2])
        iz0 = (iz-1) * (ns[1]*ns[2])
        for iy = 1:ns[2]
            iyM = mod(iy-2,ns[2]) * ns[1]
            iyP = mod(iy,  ns[2]) * ns[1]
            iy0 = (iy-1) * ns[1]
            for ix = 1:ns[1]
                # Only fastest direction contains the 1-indexing
                ixM = mod(ix-2,ns[1])+1
                ixP = mod(ix,  ns[1])+1
                ix0 = ix
                e_index = ix0+iy0+iz0

                # Neighbor elements are stored in -x,+x,-y,+y,-z,+z order
                e2e[1,e_index] = ixM+iy0+iz0
                e2e[2,e_index] = ixP+iy0+iz0
                e2e[3,e_index] = ix0+iyM+iz0
                e2e[4,e_index] = ix0+iyP+iz0
                e2e[5,e_index] = ix0+iy0+izM
                e2e[6,e_index] = ix0+iy0+izP

                # Face ID of this element's -x face will be this neighbor's +x face
                e2f[1,e_index] = 2
                e2f[2,e_index] = 1
                e2f[3,e_index] = 4
                e2f[4,e_index] = 3
                e2f[5,e_index] = 6
                e2f[6,e_index] = 5
            end
        end
    end

    # Initialize f2v

    # Initialize normals
    normals = zeros(Float64, DIM, N_FACES, n_elements)
    for iz = 1:ns[3]
        for iy = 1:ns[2]
            for ix = 1:ns[1]
                e_index = ix+(iy-1)*ns[1]+(iz-1)*ns[1]*ns[2]
                normals[1,1,e_index] = -1.0
                normals[1,2,e_index] = +1.0
                normals[2,3,e_index] = -1.0
                normals[2,4,e_index] = +1.0
                normals[3,5,e_index] = -1.0
                normals[3,6,e_index] = +1.0
            end
        end
    end

    mesh = Mesh(n_elements, n_vertices, min_edge_length, vertices, e2v, e2e, e2f, normals)
    return mesh
end

# Labels faces of a rectangle according to vertices
function init_f2v()
    f2v = Array{Int,2}(undef, N_FVERTICES, N_FACES)
    # Mapping assumes input vertices are CCW
    for iF = 1:N_FACES
        f2v[0,iF] = iF
        f2v[1,iF] = mod(iF, N_FACES)+1
    end
    return f2v
end

end
