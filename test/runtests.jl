using Test
using JSON3

## Setup the tests
include("setup.jl")

@testset "ModernBert.jl" begin
    # Add timing information and use minimal test suite
    @time begin
        using ModernBert
        include("minimal_test.jl")
    end
end
