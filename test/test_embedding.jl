using ModernBert: BertModel, embed
MODEL_PATH = joinpath(@__DIR__, "model", "model.onnx")

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
end