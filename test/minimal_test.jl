using Test
using ModernBert
using JSON3

# Load tokenizer once for all tests
vocab_path = joinpath(@__DIR__, "model", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found at $(vocab_path)"
tokenizer = load_modernbert_tokenizer(vocab_path)

@testset "Core Special Tokens" begin
    # Test essential special tokens only
    special_tokens = Dict(
        "[CLS]" => 50281,
        "[SEP]" => 50282,
        "[UNK]" => 50280
    )
    for (token, id) in special_tokens
        @test haskey(tokenizer.special_tokens, token)
        @test tokenizer.special_tokens[token] == id
    end
end

@testset "Basic Tokenization" begin
    # Test minimal sentence tokenization
    text = "Hello"  # Single word to minimize test complexity
    tokens = tokenize(tokenizer, text)
    @test length(tokens) > 0
    
    # Test full encoding with special tokens (minimal case)
    tokens, _, _ = encode(tokenizer, text)
    @test tokens[1] == 50281  # [CLS]
    @test tokens[end] == 50282  # [SEP]
    @test length(tokens) >= 3  # At least CLS, one word token, and SEP
    
    # Test [MASK] token handling
    text_mask = "The capital of France is [MASK]."
    tokens_mask, _, _ = encode(tokenizer, text_mask)
    @test 50284 in tokens_mask  # [MASK] token should be present
    
    # Test word boundary after punctuation
    text_punct = "Mr. O'Neill"
    tokens_punct = tokenize(tokenizer, text_punct)
    @test length(tokens_punct) > 0
    @test any(id -> haskey(tokenizer.id_to_token, id) && startswith(tokenizer.id_to_token[id], "Ġ"), tokens_punct)
end
