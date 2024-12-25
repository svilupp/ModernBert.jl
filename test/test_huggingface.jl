
using ModernBert: parse_repo_id, BertModel, BertTextEncoder, download_config_files

MODEL_PATH = joinpath(@__DIR__, "model", "model.onnx")

@testset "HuggingFace integration" begin
    # Test URL parsing
    @test_throws ArgumentError parse_repo_id("invalid-url")
    @test_throws ArgumentError parse_repo_id("https://example.com")

    repo_url = "https://huggingface.co/answerdotai/ModernBERT-base"
    @test parse_repo_id(repo_url) == "answerdotai/ModernBERT-base"

    # Test config download and model initialization
    temp_dir = mktempdir()
    try
        config_dir = download_config_files(repo_url, temp_dir)
        @test isfile(joinpath(config_dir, "config.json"))
        @test isfile(joinpath(config_dir, "tokenizer.json"))
        @test isfile(joinpath(config_dir, "tokenizer_config.json"))
        @test isfile(joinpath(config_dir, "special_tokens_map.json"))

        # Test model initialization with downloaded config
        model = BertModel(
            config_dir = config_dir,
            model_path = MODEL_PATH
        )
        @test model isa BertModel
        @test model.encoder isa BertTextEncoder
        @test haskey(model.encoder.vocab, "[CLS]")
    finally
        rm(temp_dir, recursive = true, force = true)
    end

    # Test direct model initialization with repo URL
    model = BertModel(
        repo_url = repo_url,
        model_path = MODEL_PATH
    )
    @test model isa BertModel
    @test model.encoder isa BertTextEncoder
end