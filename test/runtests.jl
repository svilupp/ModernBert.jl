using Test
using JSON3
using ModernBert
using ONNXRunTime
const ORT = ONNXRunTime

## Setup the tests
@testset "ModernBert.jl" begin
    include("setup.jl")
    include("test_tokenizer.jl")
    include("test_embedding.jl")
    include("test_huggingface.jl")
end
