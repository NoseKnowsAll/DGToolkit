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
    mul!
    # Operators
    # Constants

include("SolutionVector.jl")
include("LocalMatrix.jl")

end
