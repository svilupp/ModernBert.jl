using ModernBert, JSON3

# Check model files and configuration
println("\nChecking model files and configuration...")
model_path = joinpath(@__DIR__, "data", "model.onnx")
vocab_path = joinpath(@__DIR__, "data", "tokenizer.json")
@assert isfile(model_path) "model.onnx not found"
@assert isfile(vocab_path) "tokenizer.json not found"

# Load and compare tokenizer configuration
println("\nLoading and comparing tokenizer configuration...")
config = JSON3.read(read(vocab_path))
println("Vocabulary size: ", length(config["model"]["vocab"]))
special_tokens = [token["content"] for token in config["added_tokens"] if get(token, "special", false)]
println("Special tokens: ", special_tokens)
@assert length(special_tokens) >= 5 "Missing required special tokens"

# Create tokenizer
println("\nCreating tokenizer...")
tokenizer = create_tokenizer(vocab_path)

# Test tokenization
println("\nTesting tokenization...")
text = "Hello world! This is a test."
tokens = tokenize(tokenizer, text; token_ids=true, add_special_tokens=true)
println("Token IDs length: ", length(tokens))
println("Sample tokens: ", tokens[1:min(end,5)], "...")

# Test specific token sequences
println("\nTesting specific token sequences...")
@assert tokens == [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282] "Token IDs should match Python output"

text2 = "The capital of France is [MASK]."
tokens2 = tokenize(tokenizer, text2; token_ids=true, add_special_tokens=true)
@assert tokens2 == [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282] "Token IDs should match Python output for masked text"

println("\nTokenization verification complete!")
println("✓ Model files present")
println("✓ Tokenizer configuration loaded")
println("✓ Special tokens properly configured")
println("✓ Token sequences match expected output")

# Uncomment the following to test embeddings after tokenization is verified
# println("\nLoading model...")
# model = BertModel(model_path=model_path)
# embedding = embed(model, text)
# @assert length(embedding) == 1024 "Embedding dimension should be 1024"
# @assert all(isfinite, embedding) "Embeddings should be finite"



## Test cases
println("\nRunning test cases...")

# Test case 1: Basic sentence
text1 = "Hello world! This is a test."
tokens1 = tokenize(tokenizer, text1; token_ids=true, add_special_tokens=true)
println("Test 1 tokens: ", tokens1)
@assert tokens1 == [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282] "Token IDs should match Python output"
println("✓ Test case 1 passed")

# Test case 2: Masked sentence
text2 = "The capital of France is [MASK]."
tokens2 = tokenize(tokenizer, text2; token_ids=true, add_special_tokens=true)
println("Test 2 tokens: ", tokens2)
@assert tokens2 == [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282] "Token IDs should match Python output for masked text"
println("✓ Test case 2 passed")

println("\nAll tokenization tests passed!")
println("✓ Model files present")
println("✓ Tokenizer configuration loaded")
println("✓ Special tokens properly configured")
println("✓ Token sequences match expected output")
