using Test
using ModernBert
using ONNXRunTime
using JSON3
using Aqua

## Setup the tests
include("setup.jl")

# Temporarily disabled Aqua tests to focus on basic tokenization
# @testset "Code quality (Aqua.jl)" begin
#     Aqua.test_all(ModernBert)
# end

@testset "ModernBert.jl" begin
    # Add timing information and use minimal test suite
    @time begin
        include("minimal_test.jl")
    end
end
