using ModernBert: BertModel, encode, BPETokenizer

@testset "Model initialization and basic functionality" begin
    MODEL_PATH = joinpath(@__DIR__, "model", "model.onnx")
    # Test model initialization
    model = BertModel(model_path = MODEL_PATH)
    @test model isa BertModel
    @test model.session isa ONNXRunTime.InferenceSession
    @test model.encoder isa BPETokenizer

    # Test model configuration
    @test haskey(model.encoder.vocab, "[CLS]")
    @test haskey(model.encoder.vocab, "[SEP]")
    @test haskey(model.encoder.vocab, "[PAD]")
    @test model.encoder.trunc == 8192

    # Test basic functionality
    text = "Hello, world!"
    token_ids, token_type_ids, attention_mask = encode(model, text)
    @test length(token_ids) == length(token_type_ids)
    @test length(token_ids) == length(attention_mask)
    @test all(x -> x isa Integer, token_ids)
    @test all(x -> x isa Integer, token_type_ids)
    @test all(x -> x in (0, 1), attention_mask)
end