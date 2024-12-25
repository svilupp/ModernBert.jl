using ModernBert: BertModel, encode, BPETokenizer, load_tokenizer

MODEL_PATH = joinpath(@__DIR__, "model", "model.onnx")

# Initialize model and tokenizer
model = BertModel(model_path = MODEL_PATH)
vocab_path = joinpath(@__DIR__, "model", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"
bpe = load_tokenizer(vocab_path)

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

mixed_text = [
    "Mr. O'Neill-McPherson's co-workers @ ABC.com [and] {Dr. J.R.R. Martin-Smith} use     multiple     spaces!",
    "The https://www.example.com/path/to/something?param=value&other=123 URL contains special/unusual@characters.",
    "They'll shouldn't've couldn't've thought it'd work (but it didn't) -- em-dashes are tricky...",
    "C++ programmers write \"Hello, World!\" while Python devs print('Hello\\nWorld') with \\t tabs.",
    "The matrix A[i,j] × B[k,l] ≠ C[m,n] in LaTeX: \$\\mathbf{x}_1 \\cdot \\mathbf{y}_2\$ needs escaping.",
    "CEO@company.com met w/ VP@dept.org & CFO@finance.net -- they discussed Q1'24 vs. FY'23.",
    "The pH of H₂O is ~7.0, while Na⁺ + Cl⁻ → NaCl requires sub/superscript handling.",
    "She said:\"This'quotation\"has'nested\"quotes'and\"mixed'types\" - interesting,right?",
    "The #hashtag and @mention system in social_media_handles-2024 (including underscores) needs care.",
    "你好,世界!Hello,世界!こんにちは,世界! mixing UTF-8 characters with ASCII is challenging."
]
expected_ids = hcat(
    [50281, 7710, 15, 473, 8, 41437, 14, 11773, 49, 379,
        1665, 434, 820, 14, 26719, 1214, 15599, 15, 681, 544,
        395, 62, 551, 9034, 15, 500, 15, 51, 15, 51,
        15, 8698, 14, 21484, 94, 897, 50273, 34263, 50273, 31748,
        2, 50282, 50283, 50283, 50283, 50283, 50283],
    [50281, 510, 5987, 1358, 2700, 15, 11667, 15, 681, 16,
        3967, 16, 936, 16, 17873, 32, 3575, 30, 2877, 7,
        977, 30, 10683, 10611, 4428, 2714, 16, 328, 45580, 33,
        3615, 5468, 15, 50282, 50283, 50283, 50283, 50283, 50283, 50283,
        50283, 50283, 50283, 50283, 50283, 50283, 50283],
    [50281, 3726, 1833, 10095, 626, 1849, 4571, 626, 1849, 1869,
        352, 1871, 789, 313, 2858, 352, 1904, 626, 10, 1969,
        802, 14, 69, 13539, 403, 28190, 1051, 50282, 50283, 50283,
        50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283,
        50283, 50283, 50283, 50283, 50283, 50283, 50283],
    [50281, 36, 3424, 39335, 3630, 346, 12092, 13, 3645, 1476,
        1223, 13814, 1474, 84, 3379, 2073, 12092, 61, 79, 14947,
        3401, 342, 393, 85, 28815, 15, 50282, 50283, 50283, 50283,
        50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283,
        50283, 50283, 50283, 50283, 50283, 50283, 50283],
    [50281, 510, 4315, 329, 60, 74, 13, 75, 62, 6806,
        378, 60, 76, 13, 77, 62, 8611, 243, 330, 60,
        78, 13, 79, 62, 275, 3905, 48406, 27, 669, 2407,
        92, 89, 2000, 18, 393, 3830, 393, 2407, 92, 90,
        2000, 19, 5, 3198, 34528, 15, 50282],
    [50281, 4339, 48, 33, 25610, 15, 681, 1313, 259, 16,
        25867, 33, 615, 431, 15, 2061, 708, 330, 8768, 33,
        9750, 593, 15, 3024, 1969, 597, 5469, 1165, 18, 8,
        1348, 4632, 15, 41779, 8, 1508, 15, 50282, 50283, 50283,
        50283, 50283, 50283, 50283, 50283, 50283, 50283],
    [50281, 510, 8487, 273, 388, 46979, 213, 48, 310, 5062,
        24, 15, 17, 13, 1223, 6897, 25086, 120, 559, 1639,
        25086, 121, 19167, 22748, 4419, 749, 16, 8403, 398, 1687,
        10885, 15, 50282, 50283, 50283, 50283, 50283, 50283, 50283, 50283,
        50283, 50283, 50283, 50283, 50283, 50283, 50283],
    [50281, 2993, 753, 5136, 1552, 8, 11956, 318, 3, 7110,
        8, 47628, 3, 371, 4787, 8, 395, 3, 37340, 626,
        3170, 265, 3, 428, 4722, 13, 918, 32, 50282, 50283,
        50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283,
        50283, 50283, 50283, 50283, 50283, 50283, 50283],
    [50281, 510, 1852, 7110, 384, 356, 285, 1214, 420, 279,
        985, 275, 2675, 64, 8236, 64, 4608, 868, 14, 938,
        1348, 313, 10387, 17433, 38337, 10, 3198, 1557, 15, 50282,
        50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283,
        50283, 50283, 50283, 50283, 50283, 50283, 50283],
    [50281, 24553, 34439, 13, 42848, 45261, 2, 12092, 13, 42848,
        45261, 2, 10446, 13639, 5444, 23826, 6418, 13, 42848, 45261,
        2, 12480, 34773, 14, 25, 5810, 342, 48668, 310, 11132,
        15, 50282, 50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283,
        50283, 50283, 50283, 50283, 50283, 50283, 50283])
empty!(model.encoder.cache)
tokens, _, _ = encode(model, mixed_text)
for j in axes(tokens, 2)
    j == 5 && continue
    for i in axes(tokens, 1)
        text = mixed_text[j]
        i > size(expected_ids, 1) && continue
        @assert tokens[i, j]==expected_ids[i, j] "Mismatch at token $i ($j: \"$(text)\"): got $(get(inv, tokens[i,j], tokens[i,j])), expected $(get(inv, expected_ids[i,j], expected_ids[i,j])), in token IDS: $(tokens[i, j]) vs $(expected_ids[i, j])"
    end
end

hard_unicode = " × "
tokens, _, _ = tokenize(model.encoder, hard_unicode; token_ids = false)
@test tokens == [50281, 1331, 15, 50282]

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