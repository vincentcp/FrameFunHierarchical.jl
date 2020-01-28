module TestCoefficients
using Test, Hierarchical

using Hierarchical.HierarchicalIndexBase: isrefinable, finest_level,
    find_hierarchical_indices!, root_hlindex, rootblock, firstblock, level, parent,
    isrootlevel
using Hierarchical.HierarchicalCoefficients: allocindex
@testset "isrefinable" begin
    m = falses(4,4)
    m[2,1] = true
    @test isrefinable(m, (2,1), (1,1), CartesianIndex(1,1), (2,2))
    @test !isrefinable(m, (2,2), (1,1), CartesianIndex(1,1), (2,2))
    @test !isrefinable(m, (1,2), (1,1), CartesianIndex(1,1), (2,2))

    m = falses(4,4)
    m[1,2] = true
    @test !isrefinable(m, (2,1), (1,1), CartesianIndex(1,1), (2,2))
    @test !isrefinable(m, (2,2), (1,1), CartesianIndex(1,1), (2,2))
    @test isrefinable(m, (1,2), (1,1), CartesianIndex(1,1), (2,2))

    m = falses(4,4)
    m[3,2] = true
    @test !isrefinable(m, (2,1), (1,1), CartesianIndex(1,1), (2,2))
    @test !isrefinable(m, (2,2), (1,1), CartesianIndex(1,1), (2,2))
    @test !isrefinable(m, (1,2), (1,1), CartesianIndex(1,1), (2,2))
    @test !isrefinable(m, (2,1), (1,1), CartesianIndex(2,1), (2,2))
    @test !isrefinable(m, (2,2), (1,1), CartesianIndex(2,1), (2,2))
    @test isrefinable(m, (1,2), (1,1), CartesianIndex(2,1), (2,2))
end
@testset "finest_level" begin
    m = falses(10,10)
    @test finest_level(m, (1,1), CartesianIndex(1,1), (2,2)) == (1,1)
    m = trues(10,10)
    @test finest_level(m, (1,1), CartesianIndex(1,1), (2,2)) == (2,2)
    m = falses(20,20)
    m[1:4,1:4] .= true
    @test finest_level(m, (1,1), CartesianIndex(1,1), (3,3))  == (3,3)
    m = falses(20,20)
    m[1:2:4,1:2:4] .= true
    @test finest_level(m, (1,1), CartesianIndex(1,1), (3,3))  == (2,2)
    m = falses(20,20)
    m[1:1:4,1:2:4] .= true
    @test finest_level(m, (1,1), CartesianIndex(1,1), (3,3))  == (3,2)

    m = falses(10,10)
    @test finest_level(m, (1,1), CartesianIndex(2,3), (2,2)) == (1,1)
    m = trues(10,10)
    @test finest_level(m, (1,1), CartesianIndex(2,3), (2,2)) == (2,2)
    m = falses(20,20)
    m[5:8,9:12] .= true
    @test finest_level(m, (1,1), CartesianIndex(2,3), (3,3))  == (3,3)
    m = falses(20,20)
    m[5:2:8,9:2:12] .= true
    @test finest_level(m, (1,1), CartesianIndex(2,3), (3,3))  == (2,2)
    m = falses(20,20)
    m[5:1:8,9:2:12] .= true
    @test finest_level(m, (1,1), CartesianIndex(2,3), (3,3))  == (3,2)
end

@testset "find_hierarchical_indices!, root_hlindex" begin
    m = falses(4,4)
    dict = find_hierarchical_indices!(m, (2,2))
    @test dict[((0,0),CartesianIndex(1,1))] == root_hlindex((2,2),Int)


    m = trues(6,6)
    dict = find_hierarchical_indices!(m, (2,2))
    for i in CartesianIndices((3,3))
        @test dict[((1,1),i)] == HLIndices((2,2), (i.I.-1).*2 .+ 1, (i.I.-1).*2 .+2, ((0,0),CartesianIndex(1,1)), i)
    end

    m = trues(100,100)
    dict = find_hierarchical_indices!(m, (2,2))

    for i in CartesianIndices((50,50))
        @test dict[((1,1),i)] == HLIndices((2,2), (i.I.-1).*2 .+ 1, (i.I.-1).*2 .+2, ((0,0),CartesianIndex(1,1)), i)
    end


    @test isrootlevel(root_hlindex((10,10)))
end

@testset "HierarchicalIndices(trues(6,6), (2,2))" begin

    trues(6,6) isa AbstractArray{Bool,2}
    H = HierarchicalIndices(trues(6,6), (2,2));
    @test length(H) == 36
    @test rootblock(H)==root_hlindex((3,3))
    @test first(H) == ((2,2),CartesianIndex(1,1))
    @test last(H) == ((2,2),CartesianIndex(6,6))
    @test firstblock(H) == HLIndices((2,2), (1,1), (2,2), ((0,0),CartesianIndex(1,1)),CartesianIndex(1,1))
    @test isrootlevel(rootblock(H))
    @test isrootlevel(level(rootblock(H)))
    @test isrootlevel(first(rootblock(H)))
    @test level(firstblock(H)) == (2,2)
    @test !isrootlevel(firstblock(H))

    @test parent(H, firstblock(H)) == rootblock(H)
    @test parent(H, rootblock(H))==nothing
    M = zeros(Int, 6,6)
    for i in H
        @test i[1] == (2,2)
        M[i[2]] += 1
    end
    @test all(M .== 1)
end
@testset "HierarchicalIndices other" begin
    m = falses(6,6)
    m[1:2:6,1:2:6] .= true
    m[3:4,5:6] .= true
    H = HierarchicalIndices(m, (2,2))
    @test sum(m)==length(H)
    @test firstblock(H) == rootblock(H)

    ref = [((1,1), CartesianIndex(1,1)),
        ((1,1), CartesianIndex(2,1)),
        ((1,1), CartesianIndex(3,1)),
        ((1,1), CartesianIndex(1,2)),
        ((1,1), CartesianIndex(2,2)),
        ((1,1), CartesianIndex(3,2)),
        ((1,1), CartesianIndex(1,3)),
        ((2,2), CartesianIndex(3,5)),
        ((2,2), CartesianIndex(4,5)),
        ((2,2), CartesianIndex(3,6)),
        ((2,2), CartesianIndex(4,6)),
        ((1,1), CartesianIndex(3,3)),]

    for (i,j) in enumerate(H)
        @test ref[i]==j
    end

    m = falses(8,8);
    m[1:4:8,1:4:8] .= true;
    m[3:4,5:5] .= true;
    m[1:2:4,5:2:8] .= true;
    H = HierarchicalIndices(m, (3,3))

    ref = [
        ((1,1), CartesianIndex(1,1)),
        ((1,1), CartesianIndex(2,1)),

        ((2,2), CartesianIndex(1,3)),

        ((3,2), CartesianIndex(3,3)),
        ((3,2), CartesianIndex(4,3)),

        ((2,2), CartesianIndex(1,4)),
        ((2,2), CartesianIndex(2,4)),

        ((1,1), CartesianIndex(2,2))]

    for (i,j) in enumerate(H)
        @test ref[i]==j
    end

    m = falses(8,8)
        m[1:4:8,1:4:8] .= true
        m[1:2:4,5:2:8] .= true
        m[5:2:8,1:2:4] .= true
        m[5:1:8,5:1:8] .= true
        m[3:4,5:6] .= true
        m[5:6,1:4] .= true
    H = HierarchicalIndices(m, (3,3));
    @test sum(m)==length(H)
    @test firstblock(H) == rootblock(H)

    ref = [
        ((1,1), CartesianIndex(1,1)),
        ((3,3), CartesianIndex(5,1)),
        ((3,3), CartesianIndex(6,1)),
        ((3,3), CartesianIndex(5,2)),
        ((3,3), CartesianIndex(6,2)),
        ((2,2), CartesianIndex(4,1)),
        ((3,3), CartesianIndex(5,3)),
        ((3,3), CartesianIndex(6,3)),
        ((3,3), CartesianIndex(5,4)),
        ((3,3), CartesianIndex(6,4)),
        ((2,2), CartesianIndex(4,2)),
        ((2,2), CartesianIndex(1,3)),
        ((3,3), CartesianIndex(3,5)),
        ((3,3), CartesianIndex(4,5)),
        ((3,3), CartesianIndex(3,6)),
        ((3,3), CartesianIndex(4,6)),
        ((2,2), CartesianIndex(1,4)),
        ((2,2), CartesianIndex(2,4)),
        ((3,3), CartesianIndex(5,5)),
        ((3,3), CartesianIndex(6,5)),
        ((3,3), CartesianIndex(7,5)),
        ((3,3), CartesianIndex(8,5)),
        ((3,3), CartesianIndex(5,6)),
        ((3,3), CartesianIndex(6,6)),
        ((3,3), CartesianIndex(7,6)),
        ((3,3), CartesianIndex(8,6)),
        ((3,3), CartesianIndex(5,7)),
        ((3,3), CartesianIndex(6,7)),
        ((3,3), CartesianIndex(7,7)),
        ((3,3), CartesianIndex(8,7)),
        ((3,3), CartesianIndex(5,8)),
        ((3,3), CartesianIndex(6,8)),
        ((3,3), CartesianIndex(7,8)),
        ((3,3), CartesianIndex(8,8)),]

    for (i,j) in enumerate(H)
        @test ref[i]==j
    end

end

@testset "HierarchicalIndexVector" begin
    H = HierarchicalIndices(trues(6,6), (2,2))
    v = HierarchicalIndexVector(H)
    @test collect(H) == v

    m = falses(6,6)
    m[1:2:6,1:2:6] .= true
    m[3:4,5:6] .= true
    H = HierarchicalIndices(m, (2,2))
    v = HierarchicalIndexVector(H)
    @test collect(H) == v
    for (i,h) in enumerate(H)
        @test v[h] == i
    end

    m = falses(8,8);
    m[1:4:8,1:4:8] .= true;
    m[3:4,5:5] .= true;
    m[1:2:4,5:2:8] .= true;
    H = HierarchicalIndices(m, (3,3))
    v = HierarchicalIndexVector(H)
    @test collect(H) == v
    for (i,h) in enumerate(H)
        @test v[h] == i
    end

    m = falses(8,8)
        m[1:4:8,1:4:8] .= true
        m[1:2:4,5:2:8] .= true
        m[5:2:8,1:2:4] .= true
        m[5:1:8,5:1:8] .= true
        m[3:4,5:6] .= true
        m[5:6,1:4] .= true
    H = HierarchicalIndices(m, (3,3))
    v = HierarchicalIndexVector(H)
    @test collect(H) == v
    v[v[2]]
    for (i,h) in enumerate(H)
        @test v[h] == i
    end
end

@testset "HierarchicalCoefficient" begin

    m = falses(8,8);
        m[1:4:8,1:4:8] .= true;
        m[3:4,5:5] .= true;
        m[1:2:4,5:2:8] .= true;
        H = HierarchicalIndices(m, (3,3))
        vec = HierarchicalIndexVector(H)
    C = HierarchicalCoefficient{Float64}(undef, vec)

    @test allocindex(C,1) == (1,5)
    @test allocindex(C,2) == (1,6)
    @test allocindex(C,3) == (1,1)
    @test allocindex(C,4) == (2,1)
    @test allocindex(C,5) == (2,2)
    @test allocindex(C,6) == (1,3)
    @test allocindex(C,7) == (1,4)
    @test allocindex(C,8) == (1,8)

    fill!(C,1)
    @test sum(C) == length(C)

    for h in H
        @test C[h]==1
    end
end
end
