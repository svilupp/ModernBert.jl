using Test
using ModernBert
using ONNXRunTime
using JSON3
using Aqua

## Setup the tests
include("setup.jl")

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(ModernBert)
end

@testset "ModernBert.jl" begin
    # Include all test files
    include("test_model.jl")
    include("test_embedding.jl")
    include("test_tokenizer.jl")
    include("test_huggingface.jl")
end
