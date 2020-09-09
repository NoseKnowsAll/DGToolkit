"""
    module FEIO
All file input/output functionality. Includes 3DG interoperability
"""
module FEIO

using FELinearAlgebra

export dgtraits, read_array, write_array, read_solution, write_solution

"""
    dgtraits(a::Array{T,N}) where {T,N}
Compatibility of types with 3DG
"""
dgtraits(a::Array{T,N}) where {T,N} = dgtraits(eltype(a))
function dgtraits(T::Type)
    println(T)
    if T <: Bool
        return 2
    elseif T <: Integer
        return 0
    elseif T <: AbstractFloat # 3DG technically assumes C++ double
        return 1
    else
        error("Unknown DG Type")
    end
end
function dgtraits(val::Integer)
    if val == 0
        return Int64
    elseif val == 1
        return Float64
    elseif val == 2
        return Bool
    else
        error("Unknown DG Type")
    end
end

"""
    read_array(filename::AbstractString)
Read file created by `write_array()` and return the array represented by data
"""
function read_array(filename::AbstractString)
    file = open(filename, "r")
    array = read_array(file)
    close(file)
    return array
end
function read_array(io::IOStream)
    dim = read(io, Int64)
    size = Array{Int64,1}(undef, dim)
    read!(io, size)
    type = read(io, Int64)
    array = Array{dgtraits(type), dim}(undef, size...)
    read!(io, array)
    return array
end

"""
    write_array(filename::AbstractString, a)
Write the array to a specified filename in 3DG compatible format
"""
function write_array(filename::AbstractString, a)
    file = open(filename, "w")
    write_array(file, a)
    close(file)
end
function write_array(io::IOStream, a)
    if typeof(a) == SolutionVector
        a = a.data
    end
    sz = size(a)
    write(io, length(sz))
    write(io, sz...)
    write(io, dgtraits(a))
    write(io, a)
end

"""
    read_solution(filename, mesh)
Read file created by `write_solution()` and return SolutionVector defined
on a given mesh 3DG compatible format
"""
function read_solution(filename::AbstractString, mesh)
    # TODO: Will need to use mesh for parallel solutions
    data = read_array(filename)
    return SolutionVector(data)
end

"""
    write_solution(filename, u, mesh)
Write the solution defined on a given mesh to a specified filename in
3DG compatible format
"""
function write_solution(filename::AbstractString, u, mesh)
    # TODO: Will need to use mesh for parallel solutions
    write_array(filename, u.data)
end


end # module IO
