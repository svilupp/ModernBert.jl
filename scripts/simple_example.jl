
using ModernBert

## Download the model and tokenizer
model_path = download_model(
    "https://huggingface.co/answerdotai/ModernBERT-large", "data", "model.onnx")
tokenizer_path = joinpath(dirname(model_path), "tokenizer.json")

## Load the tokenizer
enc = ModernBertEncoder(tokenizer_path)

encode(enc, "hello world")

tokenize(enc, "hello world")

decode(enc, encode(enc, "hello world"))

text = "The capital of France is [MASK]."
tokenize(enc, text)
encode(enc, text)
decode(enc, encode(enc, text))

text = " [MASK] [PAD] "
tokenize(enc, text)