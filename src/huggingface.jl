module ModernBertHuggingFace

export download_config_files

using Downloads

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
            rm(file, force=true)
        end
        rethrow(e)
    end
    return target_dir
end

end # module ModernBertHuggingFace
