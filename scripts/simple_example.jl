
using ModernBert

## Download the model and tokenizer
model_path = download_model(
    "https://huggingface.co/answerdotai/ModernBERT-large", "data", "model.onnx")
tokenizer_path = joinpath(dirname(model_path), "tokenizer.json")

## Load the tokenizer
enc = ModernBertEncoder(tokenizer_path)

tokens = tokenize(enc, "hello world")
encode(enc, "hello world")
encode(enc, ["hello world", "Very long sentence with more tokens"])

model = BertModel(; model_path)
embed(model, "hello world")
texts = ["hello world", "Very long sentence with more tokens"]
embed(model, texts)