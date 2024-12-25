MODEL_PATH = joinpath(@__DIR__, "model", "model.onnx")

# Initialize model and tokenizer
model = BertModel(model_path = MODEL_PATH)
vocab_path = joinpath(@__DIR__, "model", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"
bpe = create_bpe_tokenizer(vocab_path)

@testset "Vocabulary and Special Tokens" begin
    # Test vocabulary loading
    @test length(bpe.vocab) > 0

    # Test special tokens
    special_tokens = ["[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"]
    for token in special_tokens
        @test haskey(bpe.special_tokens, token)
    end

    # Test special token IDs
    @test bpe.special_tokens["[CLS]"] == 50281
    @test bpe.special_tokens["[SEP]"] == 50282
    @test bpe.special_tokens["[MASK]"] == 50284
    @test bpe.special_tokens["[PAD]"] == 50283
    @test bpe.special_tokens["[UNK]"] == 50280
end

@testset "Basic Tokenization" begin
    # Test basic sentence
    text1 = "The capital of France is [MASK]."
    expected_ids1 = [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282]
    tokens1, _, _ = encode(model, text1)
    @test tokens1 == expected_ids1

    # Test another basic sentence
    text2 = "Hello world! This is a test."
    expected_ids2 = [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282]
    tokens2, _, _ = encode(model, text2)
    @test tokens2 == expected_ids2

    ## Multiple strings
    texts = ["Hello, world!", "This is a test.", "Multiple strings work."]
    expected_ids = hcat(
        [50281, 12092, 13, 1533, 2, 50282, 50283],
        [50281, 1552, 310, 247, 1071, 15, 50282],
        [50281, 29454, 11559, 789, 15, 50282, 50283])
    tokens, _, _ = encode(model, texts)
    @test tokens == expected_ids
end

@testset "Special Cases" begin
    # Test special token handling
    for token in ["[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"]
        result, _, _ = encode(model, token)
        @test length(result) == 3
        @test result[2] == model.encoder.vocab[token]
    end

    # Test sequence length handling
    long_text = repeat("very long text ", 100)
    long_ids, long_type_ids, long_mask = encode(model, long_text)
    @test length(long_ids) <= 512
    @test length(long_ids) == length(long_type_ids) == length(long_mask)

    # Test whitespace handling
    whitespace_text = "   "
    ws_tokens, _, _ = encode(model, whitespace_text)
    @test ws_tokens == [50281, 50275, 50282]

    # Test empty string
    empty_tokens, _, _ = encode(model, "")
    @test length(empty_tokens) == 2
    @test empty_tokens[1] == model.encoder.vocab["[CLS]"]
    @test empty_tokens[2] == model.encoder.vocab["[SEP]"]
end

@testset "Edge Cases" begin
    # Test special characters
    special_chars = "!@#%^&*()"  # Removed $ to fix linter error
    special_tokens = bpe(special_chars; token_ids = false)
    @test !isempty(special_tokens)

    # Test Unicode
    unicode_text = "Hello 世界"
    unicode_tokens = bpe(unicode_text; token_ids = false)
    @test !isempty(unicode_tokens)
end

@testset "Token ID Consistency" begin
    text = "The quick brown fox jumps over the lazy dog"

    # Get tokens and IDs separately
    tokens = bpe(text; token_ids = false)
    token_ids = bpe(text; token_ids = true)

    # Manual ID lookup
    manual_ids = Int[]
    for token in tokens
        id = get(bpe.special_tokens, token, nothing)
        if isnothing(id)
            id = get(bpe.vocab, token, bpe.special_tokens["[UNK]"])
        end
        push!(manual_ids, id)
    end

    @test token_ids == manual_ids
end

@testset "Complex Tokenization" begin
    # Test complex case with long words
    text3 = "The antidisestablishmentarianistically-minded researcher hypothesized about pseudoscientific phenomena."
    tokens3, _, _ = encode(model, text3)
    @test tokens3 == [50281, 510, 1331, 30861, 15425, 8922, 6656, 18260, 14, 23674,
        22780, 24045, 670, 10585, 5829, 850, 692, 16958, 15, 50282]

    # Test properties of complex tokenization
    @test tokens3[1] == model.encoder.vocab["[CLS]"]
    @test tokens3[end] == model.encoder.vocab["[SEP]"]

    # Test subword tokenization
    text4 = "unbelievable"
    tokens4, _, _ = encode(model, text4)
    @test tokens4 == [50281, 328, 21002, 17254, 50282]
end