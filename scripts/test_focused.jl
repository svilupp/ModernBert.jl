using Test
using ModernBert

# Load tokenizer
vocab_path = joinpath(@__DIR__, "..", "data", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"
tokenizer = load_modernbert_tokenizer(vocab_path)

# Test [MASK] token handling
function test_mask_token()
    text = "The capital of France is [MASK]."
    tokens, _, _ = encode(tokenizer, text)
    @test 50284 in tokens  # [MASK] token should be present
end

# Test word boundary after punctuation
function test_word_boundary()
    text = "Mr. O'Neill"
    tokens = tokenize(tokenizer, text)
    @test length(tokens) > 0
    @test any(id -> haskey(tokenizer.id_to_token, id) && startswith(tokenizer.id_to_token[id], "Ġ"), tokens)
end

# Run focused tests with timing
println("Running focused tests...")
@time begin
    @testset "Focused Tests" begin
        test_mask_token()
        test_word_boundary()
    end
end
