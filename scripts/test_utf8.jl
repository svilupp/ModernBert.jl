using Test
using ModernBert

# Load tokenizer
vocab_path = joinpath(@__DIR__, "..", "data", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"
tokenizer = load_modernbert_tokenizer(vocab_path)

# Test UTF-8 handling with mixed text
@testset "UTF-8 Character Handling" begin
    # Test mixed ASCII and UTF-8
    text = "你好,世界!Hello,世界!こんにちは,世界!"
    tokens, _, _ = encode(tokenizer, text)
    @test length(tokens) > 0  # Should tokenize without error
    
    # Test special token with UTF-8
    text_special = "The matrix A[i,j] × B[k,l] ≠ C[m,n]"
    tokens_special, _, _ = encode(tokenizer, text_special)
    @test length(tokens_special) > 0  # Should handle UTF-8 operators
    
    # Test boundary conditions
    text_boundary = " × "  # UTF-8 character with spaces
    tokens_boundary = tokenize(tokenizer, text_boundary)
    @test length(tokens_boundary) > 0  # Should handle UTF-8 with spaces
end

println("Running UTF-8 tests...")
@time begin
    include("test_utf8.jl")
end
