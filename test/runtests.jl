using Test
using Aqua
using ModernBert

# Run quality assurance tests
@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(ModernBert)
end

@testset "ModernBert.jl" begin
    # Test model and config loading
    @testset "Model initialization" begin
        model = ModernBertModel()
        @test model isa ModernBertModel
        @test model.session isa ONNXRunTime.InferenceSession
        @test model.encoder isa BertTextEncoder
        @test haskey(model.encoder.vocab, "[CLS]")
        @test haskey(model.encoder.vocab, "[SEP]")
        @test haskey(model.encoder.vocab, "[PAD]")
        @test model.encoder.trunc == 512
    end

    # Test single string embedding
    @testset "Single string embedding" begin
        model = ModernBertModel()
        text = "Hello, world!"
        embedding = embed(model, text)
        @test size(embedding, 2) == 1024
        @test eltype(embedding) == Float32
        token_ids, _, _ = encode(model, text)
        @test size(embedding, 1) == length(token_ids)
    end

    # Test multiple strings embedding
    @testset "Multiple strings embedding" begin
        model = ModernBertModel()
        texts = ["Hello, world!", "This is a test.", "Multiple strings work."]
        embeddings = embed(model, texts)
        @test size(embeddings, 3) == 1024
        @test size(embeddings, 2) == length(texts)
        @test eltype(embeddings) == Float32
    end

    # Test tokenization and special tokens
    @testset "Tokenization" begin
        model = ModernBertModel()
        text = "Hello, world!"
        token_ids, token_type_ids, attention_mask = encode(model, text)

        @test length(token_ids) == length(token_type_ids) == length(attention_mask)
        @test all(x -> x isa Integer, token_ids)
        @test all(x -> x isa Integer, token_type_ids)
        @test all(x -> x isa Integer, attention_mask)
        @test all(x -> x in (0, 1), attention_mask)

        vocab = model.encoder.vocab
        @test token_ids[1] == vocab["[CLS]"]
        @test token_ids[end] == vocab["[SEP]"]

        long_text = repeat("very " * text, 100)
        long_ids, long_type_ids, long_mask = encode(model, long_text)
        @test length(long_ids) <= 512
    end

    # Test BPE tokenization
    @testset "BPE tokenization" begin
        model = ModernBertModel(
            config_dir=joinpath(@__DIR__, "model"),
            model_path=joinpath(@__DIR__, "model", "model_int8.onnx")
        )
        
        # Test basic tokenization
        text1 = "The capital of France is [MASK]."
        expected_ids1 = [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282]
        tokens1, _, _ = encode(model, text1)
        @test tokens1 == expected_ids1 "Basic sentence tokenization failed"
        
        # Test another basic sentence
        text2 = "Hello world! This is a test."
        expected_ids2 = [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282]
        tokens2, _, _ = encode(model, text2)
        @test tokens2 == expected_ids2 "Basic sentence tokenization failed"
        
        # Test subword tokenization
        text3 = "unbelievable"
        tokens3, _, _ = encode(model, text3)
        token_strs = [get(Dict(v => k for (k, v) in model.encoder.vocab), id, "[UNK]") for id in tokens3]
        filtered_tokens = filter(t -> !startswith(t, "["), token_strs)
        
        # BPE specific tests
        @test length(filtered_tokens) > 1 "Word should be split into subwords"
        @test all(!startswith.(filtered_tokens[2:end], "##")) "BPE tokens should not use WordPiece '##' prefix"
        
        # Test special token handling
        special_tokens = ["[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"]
        for token in special_tokens
            result, _, _ = encode(model, token)
            @test length(result) == 3 "Special token should be encoded as [CLS] token [SEP]"
            @test result[2] == model.encoder.vocab[token] "Special token not encoded correctly"
        end
        
        # Test whitespace handling
        whitespace_text = "   "
        ws_tokens, _, _ = encode(model, whitespace_text)
        @test length(ws_tokens) > 2 "Whitespace should be properly tokenized"
        
        # Test empty string
        empty_tokens, _, _ = encode(model, "")
        @test length(empty_tokens) == 2 "Empty string should return [CLS] [SEP]"
        @test empty_tokens[1] == model.encoder.vocab["[CLS]"]
        @test empty_tokens[2] == model.encoder.vocab["[SEP]"]
    end

    # Test HuggingFace integration
    @testset "HuggingFace integration" begin
        # Test URL parsing
        @test_throws ArgumentError HuggingFace.parse_repo_id("invalid-url")
        @test_throws ArgumentError HuggingFace.parse_repo_id("https://example.com")

        repo_url = "https://huggingface.co/answerdotai/ModernBERT-large"
        @test HuggingFace.parse_repo_id(repo_url) == "answerdotai/ModernBERT-large"

        # Test config download
        temp_dir = mktempdir()
        try
            config_dir = download_config_files(repo_url, temp_dir)
            @test isfile(joinpath(config_dir, "config.json"))
            @test isfile(joinpath(config_dir, "tokenizer.json"))
            @test isfile(joinpath(config_dir, "tokenizer_config.json"))
            @test isfile(joinpath(config_dir, "special_tokens_map.json"))

            # Test model initialization with downloaded config
            model = ModernBertModel(
                config_dir=config_dir,
                model_path=joinpath(@__DIR__, "..", "data", "model.onnx")
            )
            @test model isa ModernBertModel
            @test model.encoder isa BertTextEncoder
            @test haskey(model.encoder.vocab, "[CLS]")
        finally
            rm(temp_dir, recursive=true, force=true)
        end

        # Test direct model initialization with repo URL
        temp_model = ModernBertModel(
            repo_url=repo_url,
            model_path=joinpath(@__DIR__, "..", "data", "model.onnx")
        )
        @test temp_model isa ModernBertModel
        @test temp_model.encoder isa BertTextEncoder
    end
end
