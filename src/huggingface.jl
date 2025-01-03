const CONFIG_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json"
]

"""
    parse_repo_id(url::String)

Extract repository ID from HuggingFace URL.
Examples:
```julia
parse_repo_id("https://huggingface.co/answerdotai/ModernBERT-large") # returns "answerdotai/ModernBERT-large"
```
"""
function parse_repo_id(url::String)
    m = match(r"huggingface\.co/([^/]+/[^/]+)", url)
    isnothing(m) && throw(ArgumentError("Invalid HuggingFace URL: $url"))
    return m.captures[1]
end

"""
    download_config_files(repo_url::String, target_dir::String)

Download configuration files from HuggingFace repository.
Returns the target directory path where files were downloaded.

Throws:
- ArgumentError if the URL is invalid
- Downloads.RequestError if any file fails to download
"""
function download_config_files(repo_url::String, target_dir::String)
    repo_id = parse_repo_id(repo_url)
    mkpath(target_dir)

    downloaded_files = String[]
    try
        for file in CONFIG_FILES
            url = "https://huggingface.co/$(repo_id)/resolve/main/$(file)"
            target = joinpath(target_dir, file)
            Downloads.download(url, target)
            push!(downloaded_files, target)
        end
    catch e
        # Clean up any partially downloaded files
        for file in downloaded_files
            rm(file, force = true)
        end
        rethrow(e)
    end
    return target_dir
end

"""
    download_model(repo_url::String, target_dir::String="model", model_name::String="model.onnx")

Download model and configuration files from HuggingFace repository.

# Arguments
- `repo_url::String`: URL of the HuggingFace repository (e.g. "https://huggingface.co/answerdotai/ModernBERT-large")
- `target_dir::String`: Local directory to save files (default: "model")
- `model_name::String`: Name of the model file to download (default: "model.onnx")

# Returns
The target directory path where files were downloaded.

# Throws
- `ArgumentError` if the URL is invalid
- `Downloads.RequestError` if any file fails to download

# Examples
```julia
download_model("https://huggingface.co/answerdotai/ModernBERT-large","model","model.onnx")
```
"""
function download_model(repo_url::String, target_dir::String = "model",
        model_name::String = "model.onnx")
    # Download config files first
    target_dir = download_config_files(repo_url, target_dir)

    # Download model file
    repo_id = parse_repo_id(repo_url)
    model_url = "https://huggingface.co/$(repo_id)/resolve/main/onnx/$(model_name)?download=true"
    model_path = joinpath(target_dir, model_name)

    if !isfile(model_path)
        try
            Downloads.download(model_url, model_path)
            @info "Downloaded: $model_path"
        catch e
            # Clean up if model download fails
            rm(model_path, force = true)
            rethrow(e)
        end
    end

    return model_path
end
