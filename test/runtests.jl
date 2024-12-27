using Test
using JSON3
using ModernBert

## Setup the tests
@testset "ModernBert.jl" begin
    include("setup.jl")
    include("minimal_test.jl")
end
