"""
    module FELinearAlgebra
Provides Finite Element linear algebra matrices
"""
module FELinearAlgebra

import LinearAlgebra
import Base: *, +, -, ==
import Base: getindex, setindex!, size

export
    # Types
    LocalMatrix,
    SolutionVector,
    # Functions
    issymmetric,
    isdiag,
    L2_error!,
    L2_norm,
    lp_error!,
    lp_norm,
    mul!,
    true_size
    # Operators
    # Constants

include("SolutionVector.jl")
include("LocalMatrix.jl")

end
