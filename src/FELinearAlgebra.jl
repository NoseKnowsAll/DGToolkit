"""
    module FELinearAlgebra
Provides Finite Element linear algebra matrices
"""
module FELinearAlgebra

import LinearAlgebra
import Base: *, +, -, ==, \
import Base: getindex, setindex!, size

export
    # Types
    LocalMatrix,
    JacobianMatrix,
    SolutionVector,
    # Functions
    L2_error!,
    L2_norm,
    lp_error!,
    lp_norm,
    true_size

include("SolutionVector.jl")
include("LocalMatrix.jl")

end
