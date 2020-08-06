"""
    module FELinearAlgebra
Provides Finite Element linear algebra matrices
"""
module FELinearAlgebra

import Base: getindex, setindex!, size

export
    # Types
    LocalMatrix
    # Functions
    # Operators
    # Constants

include("LocalMatrix.jl")

end
