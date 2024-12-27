using Downloads

# Import specific function after ModernBert is loaded in runtests.jl
const REPO_URL = "https://huggingface.co/answerdotai/ModernBERT-base"
MODEL_URL = "https://huggingface.co/answerdotai/ModernBERT-base/resolve/main/onnx/model_int8.onnx"
## Rename the model file to model.onnx
MODEL_PATH = joinpath(@__DIR__, "model", "model.onnx")

download_config_files(REPO_URL, joinpath(@__DIR__, "model"))
if !isfile(MODEL_PATH)
    mkpath(dirname(MODEL_PATH))
    Downloads.download(MODEL_URL, MODEL_PATH)
    @info "Downloaded: $MODEL_PATH"
end
