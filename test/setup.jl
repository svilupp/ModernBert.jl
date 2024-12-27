using ModernBert: download_config_files, download_model

download_model(REPO_URL, dirname(MODEL_PATH), "model_int8.onnx");
## Rename the model file to model.onnx
mv(joinpath(dirname(MODEL_PATH), "model_int8.onnx"), MODEL_PATH)
