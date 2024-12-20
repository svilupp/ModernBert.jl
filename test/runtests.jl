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

    # Test tokenizer matches FlashRank.jl implementation
    @testset "WordPiece tokenization" begin
        model = ModernBertModel()
        text = "unbelievable"
        token_ids, _, _ = encode(model, text)
        tokens = [get(Dict(v => k for (k, v) in model.encoder.vocab), id, "[UNK]") for id in token_ids]

        filtered_tokens = filter(t -> !startswith(t, "["), tokens)
        @test any(t -> t == "un", filtered_tokens)
        @test any(t -> t == "##believe", filtered_tokens)
        @test any(t -> t == "##able", filtered_tokens)
    end
end
