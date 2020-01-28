module HierarchicalCoefficients

using ..HierarchicalIndexBase

using MacroTools: @forward

import Base: eltype, length, size, getindex, setindex!, IndexStyle

struct HierarchicalCoefficient{T,N,I,NP1} <: AbstractVector{T}
    indices     ::HierarchicalIndexVector{N,I}
    coef_dict   ::Dict{HIndex,Tuple{Int,Int}}
    coefs       ::Vector{Array{T,NP1}}
end
export HierarchicalCoefficient
function HierarchicalCoefficient{T}(undef::UndefInitializer, H::HierarchicalIndexVector{N,I}) where {T,N,I}
    indices = HierarchicalIndices(H)
    alloc_dict = Dict{NTuple{N,Int},Int}()
    for i in indices.dict
        pointer = i[1]
        block = i[2]
        s = size(block)
        si = get(alloc_dict, s, 0) + 1
        push!(alloc_dict, s=>si)
    end
    coefs = [Array{T,N+1}(undef, dicti[1]..., dicti[2]) for dicti in alloc_dict]
    i = 1
    for dicti in alloc_dict
        alloc_dict[dicti[1]] = i
        i += 1
    end

    in_alloc_dict = Dict{NTuple{N,Int},Int}()
    for dicti in alloc_dict
        push!(in_alloc_dict, dicti[1]=>1)
    end

    dict = Dict{HIndex,Tuple{Int,Int}}()
    for i in indices.dict
        pointer = i[1]
        block = i[2]
        s = size(block)
        alloc_it = alloc_dict[s]
        in_alloc_it = in_alloc_dict[s]
        in_alloc_dict[s] = in_alloc_it + prod(s)

        push!(dict, pointer=>(alloc_it, in_alloc_it-1))
    end
    HierarchicalCoefficient{T,N,I,N+1}(H, dict, coefs)
end
@forward HierarchicalCoefficient.indices size, length
IndexStyle(::HierarchicalCoefficient) = IndexLinear()

eltype(::HierarchicalCoefficient{T}) where {T} = T
setindex!(H::HierarchicalCoefficient, x, i::Int) =
    _setindex!(H, x, allocindex(H, i))
_setindex!(H, x, I) =
    setindex!(H.coefs[I[1]], x, I[2])

getindex(coefs::HierarchicalCoefficient, i::Int) =
    _getindex(coefs, allocindex(coefs, i))
getindex(coefs::HierarchicalCoefficient, i::HIndex) =
    _getindex(coefs, allocindex(coefs, coefs.indices[i]))
_getindex(coefs, I) =
    coefs.coefs[I[1]][I[2]]
function allocindex(coefs::HierarchicalCoefficient{T,N,I}, i::Int) where {T,N,I}
    pointer = (ntuple(k->zero(I),Val(N)), CartesianIndex{N}(1))
    _allocindex(coefs.indices.indices.dict, coefs.indices.index_dict, coefs.coef_dict, pointer, i)
end

function _allocindex(dict, index_dict, coef_dict, pointer, i)
    previous_value = 0
    for (it,value) in enumerate(index_dict[pointer])
        if i <= value
            new_pointer = dict[pointer][it]
            if haskey(dict, new_pointer)
                return _allocindex(dict, index_dict, coef_dict, new_pointer, i-previous_value)
            else
                I = coef_dict[pointer]
                return I[1], I[2] + it
            end
        end
        previous_value = value
    end
    error("Outside of boundaries")
end
end
