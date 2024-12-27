using ModernBert: download_config_files, download_model

# Only download tokenizer for now - we'll test tokenizer functionality first
tokenizer_path = joinpath(@__DIR__, "model", "tokenizer.json")
if !isfile(tokenizer_path)
    download_config_files(REPO_URL, dirname(tokenizer_path))
end
