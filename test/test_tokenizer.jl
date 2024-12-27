using Test
using ModernBert: ModernBertEncoder, encode, tokenize
using TextEncodeBase
using BytePairEncoding: BPEEncoder

@testset "ModernBertEncoder Tests" begin
    # Setup
    vocab_path = joinpath(@__DIR__, "model", "tokenizer.json")
    @assert isfile(vocab_path) "tokenizer.json not found"
    encoder = ModernBertEncoder(vocab_path)

    # Basic token ID verification
    # Test special token presence and IDs
    special_tokens = Dict(
        "[CLS]" => 50281,
        "[SEP]" => 50282,
        "[MASK]" => 50284,
        "[PAD]" => 50283,
        "[UNK]" => 50280
    )
    for (token, expected_id) in special_tokens
        @test encoder.special_tokens[token] == expected_id
    end

    # Basic sentence with special tokens
    tokens = tokenize(encoder, "hello world")
    @test tokens[1] == encoder.special_tokens["[CLS]"]
    @test tokens[end] == encoder.special_tokens["[SEP]"]

    # Test encoding with attention mask
    text = "The capital of France is [MASK]."
    tokens, attention_mask, token_type_ids = encode(encoder, text)
    @test length(tokens) == length(attention_mask)
    @test all(x -> x in (0, 1), attention_mask)
    @test all(x -> x in (0, 1), token_type_ids)

    # Test batch encoding
    texts = ["Hello world!", "This is another test."]
    token_matrix, attention_matrix, type_matrix = encode(encoder, texts)
    @test size(token_matrix, 2) == length(texts)
    @test size(attention_matrix) == size(token_matrix)
    @test size(type_matrix) == size(token_matrix)

    # Test handling of special characters and punctuation
    complex_text = "Mr. O'Neill-McPherson's co-workers @ ABC.com"
    tokens = tokenize(encoder, complex_text)
    @test tokens[1] == "[CLS]"
    @test tokens[end] == "[SEP]"
    @test any(t -> occursin("'", t), tokens)

    # Test whitespace handling
    whitespace_text = "hello     world"
    tokens = tokenize(encoder, whitespace_text)
    @test length(tokens) > 3  # [CLS], hello, world, [SEP] at minimum

    # Test unknown token handling
    unknown_text = "🤖 robot"
    tokens = tokenize(encoder, unknown_text)
    @test "[UNK]" in tokens

    # Test maximum sequence handling
    long_text = repeat("a ", 1000)
    tokens = tokenize(encoder, long_text)
    @test length(tokens) > 10  # Should handle long sequences

    # Test BPEEncoder compatibility
    encoded = encoder.encode("hello world")
    @test length(encoded) > 0
    decoded = encoder.decode(encoded)
    @test occursin("hello", decoded)
    @test occursin("world", decoded)
end
