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
model = BertModel(model_path=model_path)

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



## Test against Python
text = "The capital of France is [MASK]."
# Expected output from Python for comparison
expected_input_ids = [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282]
expected_attention_mask = fill(1, 9)
expected_embeddings_first10 = [0.0016870874678716063, 0.0007494644378311932, 0.002359555335715413, 0.0032332215923815966, 0.0009463689057156444, -0.000527484342455864, 0.000894805183634162, 0.001211655093356967, -0.0006980776088312268, 0.0001597352820681408]
expected_embeddings_last10 = [0.0015778853558003902, 0.0005165686598047614, 0.0003271002206020057, 0.0030848488677293062, 0.002679278142750263, 0.0005479587125591934, 0.000785652722697705, 0.0028105787932872772, 0.0015111491084098816, 0.0005413428880274296]
tokens = encode(model, text)
@assert tokens[1] == expected_input_ids
embeddings = embed(model, text)
@assert isapprox(embeddings[1:10], expected_embeddings_first10, atol=1e-4)
@assert isapprox(embeddings[end-9:end], expected_embeddings_last10, atol=1e-4)


## Another test
text = "Hello world! This is a test."
tokens = encode(model, text)
@assert tokens[1] == [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282] "Token IDs should match Python output"
embeddings = embed(model, text)
expected_embeddings_first10 = [0.0008333486039191484, 0.0036108146887272596, 0.002033739350736141, 0.0017419838113710284, 0.0015846994938328862, 0.00058382423594594, -0.00013248198956716806, 0.0008147155749611557, -0.00018474584794603288, -0.001683817128650844]
expected_embeddings_last10 = [0.0006427527987398207, 0.000861767737660557, 0.00020104051509406418, 0.0021745895501226187, 0.00019541919755283743, 0.0008153636590577662, -0.00046477181604132056, 0.002769871847704053, -0.00042499456321820617, 0.002270730445161462]
@assert isapprox(embeddings[1:10], expected_embeddings_first10, atol=1e-4) "First 10 embedding values should match Python output"
@assert isapprox(embeddings[end-9:end], expected_embeddings_last10, atol=1e-4) "Last 10 embedding values should match Python output"
