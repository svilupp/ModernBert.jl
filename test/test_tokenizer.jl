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
    @test tokens[1] == "[CLS]"
    @test any(t -> t == "world" || t == "Ġworld", tokens)  # Accept both forms of "world"
    @test tokens[end] == "[SEP]"

    # Test encoding returns correct token IDs
    text = "The capital of France is [MASK]."
    token_ids = encode(encoder, text)
    @test token_ids[1] == encoder.special_tokens["[CLS]"]  # First token should be [CLS]
    @test encoder.special_tokens["[MASK]"] in token_ids  # Should contain [MASK] token
    @test token_ids[end] == encoder.special_tokens["[SEP]"]  # Last token should be [SEP]

    # Test batch encoding (matching simple_example.jl interface)
    texts = ["Hello world!", "This is another test."]
    token_matrix = encode(encoder, texts)
    @test size(token_matrix, 2) == length(texts)  # Each column is a sequence
    
    # Check special token positioning
    cls_id = encoder.special_tokens["[CLS]"]
    sep_id = encoder.special_tokens["[SEP]"]
    pad_id = encoder.special_tokens["[PAD]"]
    
    for col in 1:size(token_matrix, 2)
        # First token should be [CLS]
        @test token_matrix[1, col] == cls_id
        
        # Find last non-padding token
        last_content_idx = findlast(id -> id != pad_id, token_matrix[:, col])
        @test last_content_idx !== nothing
        @test token_matrix[last_content_idx, col] == sep_id  # Last content token should be [SEP]
        
        # All tokens after last_content_idx should be [PAD]
        if last_content_idx < size(token_matrix, 1)
            @test all(id -> id == pad_id, token_matrix[last_content_idx+1:end, col])
        end
    end

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
    token_ids = encode(encoder, unknown_text)
    @test encoder.special_tokens["[UNK]"] in token_ids

    # Test maximum sequence handling
    long_text = repeat("a ", 1000)
    token_ids = encode(encoder, long_text)
    @test length(token_ids) > 10  # Should handle long sequences
    @test token_ids[1] == encoder.special_tokens["[CLS]"]
    @test token_ids[end] == encoder.special_tokens["[SEP]"]

    # Test special token ID handling
    special_text = "[MASK] is a special token"
    token_ids = encode(encoder, special_text)
    @test encoder.special_tokens["[MASK]"] in token_ids
    @test token_ids[1] == encoder.special_tokens["[CLS]"]
    @test token_ids[end] == encoder.special_tokens["[SEP]"]

    # Test encode/decode compatibility (matching simple_example.jl interface)
    text = "hello world"
    tokens = tokenize(encoder, text)  # First test tokenize produces tokens
    @test all(t -> isa(t, String), tokens)  # All outputs should be strings
    @test tokens[1] == "[CLS]"
    @test tokens[end] == "[SEP]"
    
    token_ids = encode(encoder, text)  # Then test encode produces token IDs
    @test all(t -> isa(t, Integer), token_ids)  # All outputs should be integers
    @test token_ids[1] == encoder.special_tokens["[CLS]"]
    @test token_ids[end] == encoder.special_tokens["[SEP]"]
    
    decoded = encoder.decode(token_ids)
    @test occursin("hello", decoded)
    @test occursin("world", decoded)
end
