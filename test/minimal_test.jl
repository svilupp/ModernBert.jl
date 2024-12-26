using Test
using ModernBert
using JSON3

# Load tokenizer once for all tests
vocab_path = joinpath(@__DIR__, "model", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found at $(vocab_path)"
tokenizer = load_modernbert_tokenizer(vocab_path)

@testset "Core Special Tokens" begin
    # Test essential special tokens only
    @test tokenizer.special_tokens["[CLS]"] == 50281
    @test tokenizer.special_tokens["[SEP]"] == 50282
    @test tokenizer.special_tokens["[UNK]"] == 50280
end

@testset "Basic Tokenization" begin
    # Test minimal sentence tokenization
    text = "Hello world"
    tokens, _, _ = encode(tokenizer, text)
    @test tokens[1] == 50281  # [CLS]
    @test tokens[end] == 50282  # [SEP]
end

@testset "Unknown Token" begin
    # Test unknown token handling
    text = "xyz123"
    tokens, _, _ = encode(tokenizer, text)
    @test 50280 in tokens  # [UNK]
end
