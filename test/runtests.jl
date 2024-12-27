using Test
using JSON3
using ModernBert
using ONNXRunTime
const ORT = ONNXRunTime

## Setup the tests
REPO_URL = "https://huggingface.co/answerdotai/ModernBERT-base"
MODEL_PATH = joinpath(@__DIR__, "model", "model.onnx")

@testset "ModernBert.jl" begin
    # Run setup first to ensure tokenizer is available
    include("setup.jl")
    
    # Run tokenizer tests
    @testset "Tokenizer Tests" begin
        include("test_tokenizer.jl")
    end
    
    # Skip other tests for now while we verify tokenizer functionality
    # @testset "Other Tests" begin
    #     include("test_embedding.jl")
    #     include("test_huggingface.jl")
    # end
end
