using Test
using JSON3
using ModernBert

## Setup the tests
@time begin
    include("setup.jl")
    @testset "ModernBert.jl" begin
        include("minimal_test.jl")
    end
end
