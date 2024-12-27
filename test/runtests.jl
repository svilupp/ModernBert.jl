using Test
using JSON3
using ModernBert
using ONNXRunTime
const ORT = ONNXRunTime

## Setup the tests
REPO_URL = "https://huggingface.co/answerdotai/ModernBERT-base"
MODEL_PATH = joinpath(@__DIR__, "model", "model.onnx")

@testset "ModernBert.jl" begin
    include("setup.jl")
    include("test_tokenizer.jl")
    include("test_embedding.jl")
    include("test_huggingface.jl")
end
