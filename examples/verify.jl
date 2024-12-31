using ModernBert, JSON3, Test

## Download the model and tokenizer
model_path = download_model(
    "https://huggingface.co/answerdotai/ModernBERT-large", "model", "model.onnx")
tokenizer_path = joinpath(dirname(model_path), "tokenizer.json")

# Check model files and configuration
println("\nChecking model files and configuration...")
model_path = joinpath("model", "model.onnx")
tokenizer_path = joinpath("model", "tokenizer.json")
@assert isfile(model_path) "model.onnx not found"
@assert isfile(tokenizer_path) "tokenizer.json not found"

tokenizer = ModernBertEncoder(tokenizer_path)

# Test case 1: Basic sentence
text1 = "Hello world! This is a test."
tokens1 = tokenize(tokenizer, text1)
ids1 = encode(tokenizer, text1)
println("Test 1 tokens: ", tokens1)
println("Test 1 ids: ", ids1)
@test tokens1 ==
      ["[CLS]", "Hello", "Ġworld", "!", "ĠThis", "Ġis", "Ġa", "Ġtest", ".", "[SEP]"]
@test vec(ids1) == [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282]
println("✓ Test case 1 passed")

# Test case 2: Masked sentence
text2 = "The capital of France is [MASK]."
tokens2 = tokenize(tokenizer, text2)
ids2 = encode(tokenizer, text2)
println("Test 2 tokens: ", tokens2)
println("Test 2 ids: ", ids2)
@test tokens2 ==
      ["[CLS]", "The", "Ġcapital", "Ġof", "ĠFrance", "Ġis", " [MASK]", ".", "[SEP]"]
@test vec(ids2) == [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282]
println("✓ Test case 2 passed")

## Embedding
model = BertModel(; model_path)
emb1 = embed(model, text1)
emb2 = embed(model, text2)

# Test case 3: Check first 5 embedding values
println("\nChecking embedding values...")
@test isapprox(emb1[1:5],
    [0.0017, 0.0007, 0.0024, 0.0032, 0.0009],
    atol = 1e-4)
@test isapprox(emb2[1:5],
    [0.0008, 0.0036, 0.0020, 0.0017, 0.0016],
    atol = 1e-4)
println("✓ Test case 3 passed")
