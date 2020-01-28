module HierarchicalIndexBase

using MacroTools: @forward
using FillArrays
import Base: axes, size, length, getindex, IndexStyle, unsafe_getindex, first, last, iterate, eltype

export LevelIndex, LIndex, LI
"""
    const LevelIndex{N,I} = NTuple{N,I} where {N,I<:Integer}

Index that specifies the level of the hierarchical tree.
"""
const LevelIndex{N,I} = NTuple{N,I} where {N,I<:Integer}
const LIndex{N,I} = LevelIndex{N,I} where {N,I<:Integer}
const LI{N,I} = LevelIndex{N,I} where {N,I<:Integer}

export HierarchicalIndex, HIndex
"""
    const HierarchicalIndex{N,I<:Integer} = Tuple{LIndex{N,I},CartesianIndex{N}} where {N,I<:Integer}

Index that specifies the level (`index[1]`)and the index (`index[2]`) in that level of a hierarchical tree.
"""
const HierarchicalIndex{N,I<:Integer} = Tuple{LIndex{N,I},CartesianIndex{N}} where {N,I<:Integer}
const HIndex{N,I<:Integer} = HierarchicalIndex{N,I} where {N,I<:Integer}


struct HierarchicalLevelIndicesBase{N,I<:Integer} <: AbstractArray{HIndex{N,I},N}
    level   ::LevelIndex{N,I}
    indices ::CartesianIndices{N,NTuple{N,UnitRange{Int}}}
end
export level
level(H::HierarchicalLevelIndicesBase) = H.level
level(I::HIndex) = I[1]
index(I::HIndex) = I[2]

HierarchicalLevelIndicesBase(l::LevelIndex{N},s::NTuple{N,<:Integer}) where N =
    HierarchicalLevelIndicesBase(l,CartesianIndices(map(x->UnitRange(1:x), s)))

HierarchicalLevelIndicesBase(l::LevelIndex{N},s1::NTuple{N,<:Integer},s2::NTuple{N,<:Integer}) where N =
    HierarchicalLevelIndicesBase(l,CartesianIndices(map((x,y)->UnitRange(x:y), s1, s2)))
HierarchicalLevelIndicesBase(l::LevelIndex{N},I::NTuple{N}) where N =
    HierarchicalLevelIndicesBase(l,CartesianIndices(I))

@forward HierarchicalLevelIndicesBase.indices axes, size, length

getindex(H::HierarchicalLevelIndicesBase, i...) =
    (H.level, getindex(H.indices,i...))
first(H::HierarchicalLevelIndicesBase) =
    (H.level, first(H.indices))
last(H::HierarchicalLevelIndicesBase) =
    (H.level, last(H.indices))
function iterate(H::HierarchicalLevelIndicesBase)
    next = iterate(H.indices)
    if next == nothing
        return nothing
    end
    it, state = next
    (H.level, it), state
end
function iterate(H::HierarchicalLevelIndicesBase, state)
    next = iterate(H.indices, state)
    if next == nothing
        return nothing
    end
    it, state = next
    (H.level, it), state
end

export HierarchicalLevelIndices, HLIndices
"""
    struct HierarchicalLevelIndices{N,I<:Integer} <: AbstractArray{HIndex{N,I},N}

A container of a box of HierarchicalIndex's
# Example
```jldocs
julia> using Hierarchical
julia> HierarchicalLevelIndices((1,1),(3,3),(7,7))
5×5 HierarchicalLevelIndices{2,Int64}:
 ((1, 1), CartesianIndex(3, 3))  ((1, 1), CartesianIndex(3, 4))  ((1, 1), CartesianIndex(3, 5))  ((1, 1), CartesianIndex(3, 6))  ((1, 1), CartesianIndex(3, 7))
 ((1, 1), CartesianIndex(4, 3))  ((1, 1), CartesianIndex(4, 4))  ((1, 1), CartesianIndex(4, 5))  ((1, 1), CartesianIndex(4, 6))  ((1, 1), CartesianIndex(4, 7))
 ((1, 1), CartesianIndex(5, 3))  ((1, 1), CartesianIndex(5, 4))  ((1, 1), CartesianIndex(5, 5))  ((1, 1), CartesianIndex(5, 6))  ((1, 1), CartesianIndex(5, 7))
 ((1, 1), CartesianIndex(6, 3))  ((1, 1), CartesianIndex(6, 4))  ((1, 1), CartesianIndex(6, 5))  ((1, 1), CartesianIndex(6, 6))  ((1, 1), CartesianIndex(6, 7))
 ((1, 1), CartesianIndex(7, 3))  ((1, 1), CartesianIndex(7, 4))  ((1, 1), CartesianIndex(7, 5))  ((1, 1), CartesianIndex(7, 6))  ((1, 1), CartesianIndex(7, 7))
```
"""
struct HierarchicalLevelIndices{N,I<:Integer} <: AbstractArray{HIndex{N,I},N}
    base    ::HierarchicalLevelIndicesBase{N,I}
    # Each block holds a reference to the block above (reference has only meaning using the dict in HierarchicalIndices)
    parent_block  ::HierarchicalIndex{N,I}
    parent_block_state::CartesianIndex{N}
end
const HLIndices{N,I} = HierarchicalLevelIndices{N,I} where {N,I<:Integer}

root_hlindex(size::NTuple{N,Int}, ::Type{I}=UInt8) where {N,I} =
    HLIndices(HierarchicalLevelIndicesBase(ntuple(k->one(I),Val(N)), size), (ntuple(k->zero(I),Val(N)),CartesianIndex{N}(1)), CartesianIndex{N}(1))

HLIndices(l::LevelIndex{N},s::NTuple{N,<:Integer}, parent=(ntuple(k->1,Val(N)),CartesianIndex{N}(1)), parent_state=CartesianIndex{N}(0)) where N =
    HLIndices(HierarchicalLevelIndicesBase(l,CartesianIndices(map(x->UnitRange(1:x), s))), parent, parent_state)

HLIndices(l::LevelIndex{N},s1::NTuple{N,<:Integer},s2::NTuple{N,<:Integer}, parent=(ntuple(k->1,Val(N)),CartesianIndex{N}(1)), parent_state=CartesianIndex{N}(0)) where N =
    HLIndices(HierarchicalLevelIndicesBase(l,CartesianIndices(map((x,y)->UnitRange(x:y), s1, s2))), parent, parent_state)

@forward HierarchicalLevelIndices.base axes, size, length, getindex, level, first, last, iterate

export HierarchicalIndices
"""
    struct HierarchicalIndices{N,I} <: AbstractVector{HIndex{N,I}}

All indices of a hierarchical structure.
Here the assumption is made that refinement is recursive.
For example, A level (3,3) can not exist without a level (2,3), (3,2) or (2,2).

You can HierarchicalIndices one using a mask.
# Example
```jldocs
julia> m = falses(8,8);

julia> m[1:4:8,1:4:8] .= true;

julia> m[3:4,5:5] .= true;

julia> m[1:2:4,5:2:8] .= true;

julia> H = HierarchicalIndices(m, (3,3));

julia> collect(H)
8-element Array{Tuple{Tuple{Int64,Int64},CartesianIndex{2}},1}:
 ((1, 1), CartesianIndex(1, 1))
 ((1, 1), CartesianIndex(2, 1))
 ((2, 2), CartesianIndex(1, 3))
 ((3, 2), CartesianIndex(3, 3))
 ((3, 2), CartesianIndex(4, 3))
 ((2, 2), CartesianIndex(1, 4))
 ((2, 2), CartesianIndex(2, 4))
 ((1, 1), CartesianIndex(2, 2))

julia> H = HierarchicalIndices(m, (0x3,0x3));

julia> collect(H)
8-element Array{Tuple{Tuple{UInt8,UInt8},CartesianIndex{2}},1}:
 ((0x01, 0x01), CartesianIndex(1, 1))
 ((0x01, 0x01), CartesianIndex(2, 1))
 ((0x02, 0x02), CartesianIndex(1, 3))
 ((0x03, 0x02), CartesianIndex(3, 3))
 ((0x03, 0x02), CartesianIndex(4, 3))
 ((0x02, 0x02), CartesianIndex(1, 4))
 ((0x02, 0x02), CartesianIndex(2, 4))
 ((0x01, 0x01), CartesianIndex(2, 2))
```
"""
struct HierarchicalIndices{N,I} #<: AbstractVector{HIndex{N,I}}
    # Map from a HIndex to a set of HIndices on a finer level and the number of indices underneath
    dict :: Dict{HIndex{N,I},HLIndices{N,I}}
    L::LIndex{N,I}
end
function HierarchicalIndices(mask::AbstractArray{Bool,N}, L::LIndex{N,I}) where {N,I}
    m = copy(mask)
    dict = find_hierarchical_indices!(m, L)
    @assert sum(m) == 0
    HierarchicalIndices{N,I}(dict, L)
end

level(H::HierarchicalIndices) = H.L
rootblock(H::HierarchicalIndices{N,I}) where {N,I}=
    H.dict[(ntuple(k->zero(I),Val(N)),CartesianIndex{N}(1))]

function length(H::HierarchicalIndices)
    l = reduce(+, length(v) for v in values(H.dict))
    l - length(H.dict)+1
end
eltype(::HierarchicalIndices{N,I}) where {N,I} = HIndex{N,I}
IndexStyle(::Type{HierarchicalIndices}) = IndexLinear()

function getindex(H::HierarchicalIndices, i::Int)
    error("Not implemented")
end

function first(H::HierarchicalIndices)
    I = first(rootblock(H))
    if haskey(H.dict, I)
        return _first(H, H.dict[I])
    end
    I
end

function _first(H::HierarchicalIndices, hlixs::HLIndices)
    I = first(hlixs)
    if haskey(H.dict, I)
        return _first(H, H.dict[I])
    end
    I
end

function last(H::HierarchicalIndices)
    I = last(rootblock(H))
    if haskey(H.dict, I)
        return _last(H, H.dict[I])
    end
    I
end

function _last(H::HierarchicalIndices, hlixs::HLIndices)
    I = last(hlixs)
    if haskey(H.dict, I)
        return _last(H, H.dict[I])
    end
    I
end

function firstblock(H::HierarchicalIndices)
    block = rootblock(H)
    I = first(block)
    if haskey(H.dict, I)
        return _firstblock(H, H.dict[I])
    end
    block
end

function _firstblock(H::HierarchicalIndices, hlixs::HLIndices)
    I = first(hlixs)
    if haskey(H.dict, I)
        return _firstblock(H, H.dict[I])
    end
    hlixs
end

isrootlevel(hlixs::HLIndices) =
    all(i==1 for i in level(hlixs))
isrootlevel(hix::HierarchicalIndex) =
    isrootlevel(hix[1])
function isrootlevel(lix::LevelIndex)
    all(i==1 for i in lix)
end

function parent(H::HierarchicalIndices, block::HierarchicalLevelIndices)
    if isrootlevel(block)
        return nothing
    end
    H.dict[block.parent_block]
end



@inline function iterate(H::HierarchicalIndices)
    if length(H) > 0
        block = firstblock(H)
        block_it, block_state = iterate(block)
        (block_it, (block, block_state))
    end
end

@inline function _parent_iterate(H::HierarchicalIndices, block::HierarchicalLevelIndices)
    parent_block = parent(H, block)
    if parent_block == nothing
        return nothing
    end
    parent_block_state = block.parent_block_state
    next = iterate(parent_block, parent_block_state)
    if next == nothing
        next = _parent_iterate(H, parent_block)
        if next == nothing
            return nothing
        else
            next_block, next_block_it, next_block_state = next
        end
    else
        next_block_it, next_block_state = next
        if haskey(H.dict, next_block_it)
            next_block = _firstblock(H, H.dict[next_block_it])
            next_block_it, next_block_state = iterate(next_block)
        else
            next_block = parent_block
        end
    end
    (next_block, next_block_it, next_block_state)
end

@inline function iterate(H::HierarchicalIndices, state)
    block, block_state = state
    next = iterate(block, block_state)
    if next==nothing
        next = _parent_iterate(H, block)
        if next == nothing
            return nothing
        end
        next_block, next_block_it, next_block_state = next
    else
        next_block_it, next_block_state = next
        if haskey(H.dict, next_block_it)
            next_block = _firstblock(H, H.dict[next_block_it])
            next_block_it, next_block_state = iterate(next_block)
        else
            next_block = block
        end
    end
    (next_block_it, (next_block, next_block_state))
end

_level_size(fine_size::NTuple{N,Int}, coarse_level::LIndex, fine_level::LIndex) where N =
    div.(fine_size,_level_box_size(coarse_level, fine_level))
_level_box_size(coarse_level::LIndex, fine_level::LIndex) =
    one(Int) .<< (fine_level .- coarse_level)

function find_hierarchical_indices!(mask::AbstractArray{Bool,N}, L::LIndex{N,I}) where {N,I}
    dict = Dict{HIndex{N,I},HLIndices{N,I}}()
    root_level = ntuple(k->one(I), Val(N))
    root_block = root_hlindex(_level_size(size(mask), root_level, L), I)
    root_pointer = (ntuple(k->zero(I),Val(N)),CartesianIndex{N}(1))
    push!(dict, root_pointer=>root_block)

    for root_level_mindex in LevelIterator(size(mask), root_level, root_level, L)
        mask[root_level_mindex] = false
        start_tree!(dict, mask, root_level, root_pointer, root_block, root_level_mindex, L)
    end
    dict
end

function start_tree!(dict, mask, parent_level::LIndex{N,I}, parent_pointer::HIndex{N,I}, parent_block::HierarchicalLevelIndices{N,I}, parent_level_mindex::CartesianIndex{N}, L::LIndex{N,I}) where {N,I}
    parent_level_index = CartesianIndex(ntuple(k->div(parent_level_mindex[k]-1, 1 << (L[k]-parent_level[k])) + 1,Val(N)))
    root_level = finest_level(mask, parent_level, parent_level_index, L)
    # refinement is happening
    if any(root_level[i]>parent_level[i] for i in 1:N)
        root_size = _level_box_size(parent_level, root_level)
        root_level_index = CartesianIndex(ntuple(k->div(parent_level_mindex[k]-1, 1 << (L[k]-root_level[k])) + 1,Val(N)))
        root_pointer = (parent_level, parent_level_index)
        root_block = HLIndices(root_level, root_level_index.I, ntuple(k->root_level_index[k] + root_size[k]-1 ,Val(N)), parent_pointer, parent_level_index)
        push!(dict, root_pointer=>root_block)
        root_mindices = BlockIterator(parent_level, root_level, L, parent_level_index)

        for root_level_mindex in root_mindices
            mask[root_level_mindex] = false
            start_tree!(dict, mask, root_level, root_pointer, root_block, root_level_mindex, L)
        end
    end
    dict
end

function finest_level(mask, root_level::LIndex{N,I}, index::CartesianIndex, L::LIndex{N,I}) where {N,I}
    refined = true
    level = root_level
    while refined
        refined = false
        for i in 1:N
            if level[i]<L[i]
                # Try to refine in dimension i
                new_level = ntuple(k->k==i ? level[k]+one(I) : level[k], Val(N))
                if isrefinable(mask, new_level, root_level, index, L)
                    refined = true
                    level = new_level

                end
            end
        end
    end
    level
end

function isrefinable(mask, new_level, root_level, index, L)
    @assert new_level!=root_level

    block_it = BlockIterator(root_level, new_level, L, index)
    next = iterate(block_it)
    if next == nothing
        return false
    end
    _, state = next
    for i in Iterators.rest(block_it, state)
        if !mask[i]
            return false
        end
    end
    return true
end

struct BlockIterator{N,I}
    coarse_level::LIndex{N,I}
    fine_level::LIndex{N,I}
    L::LIndex{N,I}
    coarse_index::CartesianIndex{N}
end

function iterate(itr::BlockIterator{N,I}) where {N,I}
    # the number of indices per box on the fine level visible from the coase level
    lbs = _level_box_size(itr.coarse_level, itr.fine_level)
    # iterator over the indices from the fine level
    levelmask = CartesianIndices(lbs)
    fine_index_step = 1 .<< (itr.L .- itr.fine_level)

    fine_next = iterate(levelmask)
    if fine_next == nothing
        return nothing
    end

    fine_it, fine_state = fine_next
    (_mul(_mul(itr.coarse_index , size(levelmask))+ fine_it - CartesianIndex{N}(1),fine_index_step),
        (fine_state, levelmask, fine_index_step))
end

function iterate(itr::BlockIterator{N,I}, state) where {N,I}
    fine_state, levelmask, fine_index_step = state
    fine_next = iterate(levelmask, fine_state)
    if fine_next == nothing
        return nothing
    else
        fine_it, fine_state = fine_next
    end
    (_mul(_mul(itr.coarse_index , size(levelmask))+ fine_it - CartesianIndex{N}(1),fine_index_step),
        (fine_state, levelmask, fine_index_step))
end

struct LevelIterator{N,I}
    fine_size::NTuple{N,Int}
    coarse_level::LIndex{N,I}
    fine_level::LIndex{N,I}
    L::LIndex{N,I}
end

function iterate(itr::LevelIterator{N,I}) where {N,I}
    # the number of indices per box on the fine level visible from the coase level
    lbs = _level_box_size(itr.coarse_level, itr.fine_level)
    # Iterator at the level of the coarse indices
    coarse_index_iterator = CartesianIndices(_level_size(itr.fine_size, itr.coarse_level, itr.L))
    # iterator over the indices from the fine level
    levelmask = CartesianIndices(lbs)

    fine_index_step = 1 .<< (itr.L .- itr.fine_level)

    coarse_next = iterate(coarse_index_iterator)
    if coarse_next == nothing
        return nothing
    end
    coarse_it, coarse_state = coarse_next
    fine_next = iterate(levelmask)
    if fine_next == nothing
        return nothing
    end
    fine_it, fine_state = fine_next
    (_mul(_mul(coarse_it , size(levelmask))+ fine_it - CartesianIndex{N}(1),fine_index_step),
        (coarse_it, fine_state, coarse_state, levelmask, coarse_index_iterator, fine_index_step))
end

_mul(cart::CartesianIndex{N}, size::NTuple{N}) where N =
    CartesianIndex(ntuple(i->(size[i]*(cart[i]-1))+1, Val(N)))

function iterate(itr::LevelIterator{N,I}, state) where {N,I}
    coarse_it, fine_state, coarse_state, levelmask, coarse_index_iterator, fine_index_step = state
    fine_next = iterate(levelmask, fine_state)
    if fine_next == nothing
        coarse_next = iterate(coarse_index_iterator, coarse_state)
        if coarse_next == nothing
            return nothing
        else
            coarse_it, coarse_state = coarse_next
            fine_it, fine_state = iterate(levelmask)
        end
    else
        fine_it, fine_state = fine_next
    end
    (_mul(_mul(coarse_it , size(levelmask))+ fine_it - CartesianIndex{N}(1),fine_index_step),
        (coarse_it, fine_state, coarse_state, levelmask, coarse_index_iterator, fine_index_step))
end

export HierarchicalIndexVector
struct HierarchicalIndexVector{N,I} <: AbstractVector{HIndex{N,I}}
    # Map from a HIndex to a set of HIndices on a finer level and the number of indices underneath
    indices    :: HierarchicalIndices{N,I}
    index_dict :: Dict{HIndex{N,I},Union{Vector{Int},UnitRange{Int}}}

    function HierarchicalIndexVector(H::HierarchicalIndices{N,I}) where {N,I}
        new{N,I}(H,_index_dict(H))
    end
end

IndexStyle(::HierarchicalIndexVector) = IndexLinear()
size(H::HierarchicalIndexVector) = (length(H),)
setindex!(H::HierarchicalIndexVector, i::Int) = error("`setindex` not possible for `HierarchicalIndexVector`")
@forward HierarchicalIndexVector.indices rootblock, firstblock, first, last, level, iterate, length, eltype
HierarchicalIndices(H::HierarchicalIndexVector) =
    H.indices

function isleafblock(H::HierarchicalIndices, block::HierarchicalLevelIndices)
    for i in block
        if haskey(H.dict, i)
            return false
        end
    end
    return true
end

function childrenblocksready(block::HierarchicalLevelIndices, mask)
    for i in block
        if haskey(mask, i) && mask[i]
            return false
        end
    end
    return true
end

function _index_dict(H::HierarchicalIndices{N,I}) where {N,I}
    index_dict = Dict{HIndex{N,I},AbstractArray{Int,N}}()
    index_mask = Dict{HIndex{N,I},Bool}()

    for dicti in H.dict
        block_pointer = dicti[1]
        block = dicti[2]

        if isleafblock(H, block)
            push!(index_dict, block_pointer=>Ones{Int}(size(block)...))
            push!(index_mask, block_pointer=>false)
        else
            push!(index_dict, block_pointer=>ones(Int, size(block)))
            push!(index_mask, block_pointer=>true)
        end
    end

    while sum(values(index_mask)) > 0
        for dicti in H.dict
            block_pointer = dicti[1]
            if index_mask[block_pointer]
                block = dicti[2]
                if childrenblocksready(block, index_mask)
                    for (i,block_it) in enumerate(block)
                        if haskey(index_dict, block_it)
                            index_dict[block_pointer][i] = sum(index_dict[block_it])
                        end
                    end
                    index_mask[block_pointer] = false
                end
            end
        end
    end
    Dict{HIndex{N,I},Union{Vector{Int},UnitRange{Int}}}(dicti[1]=> _cumsum(dicti[2]) for dicti in index_dict)
end

_cumsum(A::Ones{Int}) =
    1:length(A)
_cumsum(A::Array{Int}) =
    cumsum(A[:])

function getindex(vec::HierarchicalIndexVector{N,I}, i::Int) where {N,I}
    pointer = (ntuple(k->zero(I),Val(N)), CartesianIndex{N}(1))
    _getindex(vec.indices.dict, vec.index_dict, pointer, i)
end

function _getindex(dict, index_dict, pointer::HIndex, i::Int)
    previous_value = 0
    for (it,value) in enumerate(index_dict[pointer])
        if i <= value
            new_pointer = dict[pointer][it]
            if haskey(dict, new_pointer)
                return _getindex(dict, index_dict, new_pointer, i-previous_value)
            else
                return dict[pointer][it]
            end
        end
        previous_value = value
    end
    error("$i not in bounds")
end

function getindex(vec::HierarchicalIndexVector{N,I}, i::HIndex{N,I}) where {N,I}
    block = rootblock(vec)
    cm = vec.index_dict[(ntuple(k->zero(I),Val(N)),CartesianIndex{N}(1))]
    index_level = level(i)
    L = level(vec)
    mask_index = CartesianIndex((index(i).I .- 1) .*  (1 .<< (L.-index_level)) .+ 1)
    _getindex(vec, block, cm, mask_index, CartesianIndex{N}(1), 1, i)
end

function _getindex(vec::HierarchicalIndexVector{N,I}, block::HierarchicalLevelIndices{N,I}, cm::Union{UnitRange{Int},Vector{Int}},
        mask_index::CartesianIndex{N}, prev_mask_index::CartesianIndex{N}, i::Int, j::HIndex{N,I}) where {N,I}
    block_level = level(block)
    L = level(vec)
    block_index = CartesianIndex(div.(mask_index.I .- prev_mask_index.I, 1 .<< (L.-block_level)) .+ 1)
    block_mask_index = CartesianIndex((block_index.I .- 1) .*  (1 .<< (L.-block_level)) .+ 1)
    if checkbounds(Bool, block, block_index)
        @inbounds new_index = block[block_index]
    else
        error(`$j not in this $(typeof(vec))`)
    end
    if haskey(vec.indices.dict, new_index)
        new_block = vec.indices.dict[new_index]
        new_cm = vec.index_dict[new_index]
        @inbounds nexti = LinearIndices(size(block))[block_index]-1 == 0 ?
            i : i+Int(cm[LinearIndices(size(block))[block_index]-1])

        return _getindex(vec, new_block, new_cm, mask_index, CartesianIndex(prev_mask_index.I .+ block_mask_index.I .-1), nexti, j)
    elseif  new_index == j
        if LinearIndices(size(block))[block_index]-1 == 0
            return i
        else

            return i+Int(cm[LinearIndices(size(block))[block_index]-1])
        end
    end
    error(`$j not in this $(typeof(vec))`)
end
end
