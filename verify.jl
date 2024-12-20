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

# Load model
println("\nLoading model...")
model = ModernBertModel(model_path=model_path, vocab_path=vocab_path)

# Test tokenization
println("\nTesting tokenization...")
text = "Hello world! This is a test."
tokens = encode(model, text)
println("Token IDs length: ", length(tokens[1]))
println("Sample tokens: ", tokens[1][1:min(end,5)], "...")

# Test single string embedding
println("\nTesting single string embedding...")
embedding = embed(model, text)
println("Single embedding shape: ", size(embedding))
@assert length(embedding) == 1024 "Embedding dimension should be 1024"
@assert all(isfinite, embedding) "Embeddings should be finite"

# Verify embedding normalization
embedding_norm = sqrt(sum(embedding.^2))
println("Embedding L2 norm: ", embedding_norm)
@assert 0.1 < embedding_norm < 10.0 "Embedding norm should be reasonable"

# Test batch processing
println("\nTesting batch processing...")
texts = ["First sentence.", "Second example.", "Third test case."]
embeddings = embed(model, texts)
println("Batch embeddings shape: ", size(embeddings))
@assert size(embeddings, 2) == 1024 "Batch embedding dimension should be 1024"  # Second dimension should be 1024
@assert size(embeddings, 1) == length(texts) "Number of embeddings should match number of input texts"
# Verify L2 normalization for each embedding in batch
for i in 1:size(embeddings, 1)
    embedding_norm = sqrt(sum(embeddings[i,:].^2))
    @assert 0.9 < embedding_norm < 1.1 "Batch embedding $i L2 norm should be approximately 1.0"
end

println("\nVerification complete!")
println("✓ Model files present and loaded")
println("✓ Tokenizer configuration loaded")
println("✓ Special tokens properly configured")
println("✓ Single string processing works")
println("✓ Batch processing works")
println("✓ Embedding dimensions match ModernBERT-large (1024)")
