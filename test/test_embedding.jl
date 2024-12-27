using ModernBert: BertModel, encode, ModernBertTokenizer
using ModernBert: BertModel, embed
MODEL_PATH = joinpath(@__DIR__, "model", "model.onnx")

@testset "Model initialization and basic functionality" begin
    # Test model initialization
    model = BertModel(model_path = MODEL_PATH)
    @test model isa BertModel
    @test model.session isa ONNXRunTime.InferenceSession
    @test model.tokenizer isa ModernBertTokenizer

    # Test model configuration
    @test haskey(model.tokenizer.vocab, "[CLS]")
    @test haskey(model.tokenizer.vocab, "[SEP]")
    @test haskey(model.tokenizer.vocab, "[PAD]")

    # Test basic functionality
    text = "Hello, world!"
    token_ids, token_type_ids, attention_mask = encode(model, text)
    @test length(token_ids) == length(token_type_ids)
    @test length(token_ids) == length(attention_mask)
    @test all(x -> x isa Integer, token_ids)
    @test all(x -> x isa Integer, token_type_ids)
    @test all(x -> x in (0, 1), attention_mask)
end

@testset "Text embedding functionality" begin
    model = BertModel(model_path = MODEL_PATH)

    # Test single string embedding
    text = "Hello, world!"
    embedding = embed(model, text)
    @test length(embedding) == 1024
    @test eltype(embedding) == Float32

    # Test multiple strings embedding
    texts = ["Hello, world!", "This is a test.", "Multiple strings work."]
    embeddings = embed(model, texts)
    @test size(embeddings, 2) == 1024
    @test size(embeddings, 1) == length(texts)
    @test eltype(embeddings) == Float32

    # Test empty input handling
    empty_text = ""
    empty_embedding = embed(model, empty_text)
    @test length(empty_embedding) == 1024
end;