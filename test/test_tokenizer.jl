using ModernBert: ModernBertEncoder, encode, tokenize
using BytePairEncoding: BPEEncoder

# Setup
vocab_path = joinpath(@__DIR__, "model", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"
encoder = ModernBertEncoder(vocab_path)

@testset "ModernBertEncoder Tests" begin
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
            @test all(id -> id == pad_id, token_matrix[(last_content_idx + 1):end, col])
        end
    end

    # Test handling of special characters and punctuation
    complex_text = "Mr. O'Neill-McPherson's co-workers @ ABC.com"
    tokens = tokenize(encoder, complex_text)
    @test tokens[1] == "[CLS]"
    @test tokens[end] == "[SEP]"

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
@testset "Basic Tokenization" begin
    # Test basic sentence
    text1 = "The capital of France is [MASK]."
    expected_ids1 = [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282]
    tokens1 = encode(encoder, text1)
    @test tokens1 == expected_ids1

    # Test another basic sentence
    text2 = "Hello world! This is a test."
    expected_ids2 = [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282]
    tokens2 = encode(encoder, text2)
    @test tokens2 == expected_ids2
end

@testset "Challenging Cases" begin
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
    tokens = encode(encoder, mixed_text)
    for j in axes(tokens, 2)
        for i in axes(tokens, 1)
            text = mixed_text[j]
            i > size(expected_ids, 1) && continue
            check = tokens[i, j] == expected_ids[i, j]
            check ||
                @error "Test failed: Mismatch at token $i ($j: \"$(text)\"): got $(get(encoder.id_to_token, tokens[i,j], tokens[i,j])), expected $(get(encoder.id_to_token, expected_ids[i,j], expected_ids[i,j])), in token IDS: $(tokens[i, j]) vs $(expected_ids[i, j])"
            @test tokens[i, j] == expected_ids[i, j]
        end
    end
end

@testset "Special Cases" begin
    long_text = repeat("very long text ", 100)
    long_result = encode(encoder, long_text)
    @test size(long_result, 1) <= 512

    # Test whitespace handling
    whitespace_text = "   "
    ws_result = encode(encoder, whitespace_text)
    @test ws_result == [50281, 50275, 50282]

    # Test empty string
    empty_result = encode(encoder, "")
    @test length(empty_result) == 2
    @test empty_result[1] == encoder.special_tokens["[CLS]"]
    @test empty_result[2] == encoder.special_tokens["[SEP]"]
end

@testset "Edge Cases" begin
    # Test special characters
    special_chars = "!@#%^&*()"  # Removed $ to fix linter error
    special_tokens = tokenize(encoder, special_chars)
    @test !isempty(special_tokens)

    # Test Unicode
    unicode_text = "Hello 世界"
    unicode_tokens = tokenize(encoder, unicode_text)
    @test !isempty(unicode_tokens)
end

@testset "Complex Tokenization" begin
    # Test complex case with long words
    text3 = "The antidisestablishmentarianistically-minded researcher hypothesized about pseudoscientific phenomena."
    result3 = encode(encoder, text3)
    @test result3 == [50281, 510, 1331, 30861, 15425, 8922, 6656, 18260, 14, 23674,
        22780, 24045, 670, 10585, 5829, 850, 692, 16958, 15, 50282]

    # Test properties of complex tokenization
    @test result3[1] == encoder.special_tokens["[CLS]"]
    @test result3[end] == encoder.special_tokens["[SEP]"]

    # Test subword tokenization
    text4 = "unbelievable"
    result4 = encode(encoder, text4)
    @test result4 == [50281, 328, 21002, 17254, 50282]
end
