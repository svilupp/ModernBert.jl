using ModernBert, JSON3

# Check model files and configuration
println("\nChecking model files and configuration...")
vocab_path = joinpath(@__DIR__, "data", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"

# Load and compare tokenizer configuration
println("\nLoading and comparing tokenizer configuration...")
config = JSON3.read(read(vocab_path))
println("Vocabulary size: ", length(config["model"]["vocab"]))
special_tokens = [token["content"] for token in config["added_tokens"] if get(token, "special", false)]
println("Special tokens: ", special_tokens)
@assert length(special_tokens) >= 5 "Missing required special tokens"

# Create BertTextEncoder
println("\nCreating encoder...")
bpe = create_bpe_tokenizer(vocab_path)

# Create vocabulary from config
vocab = Dict{String,Int}()
for (token, id) in config["model"]["vocab"]
    vocab[String(token)] = id
end

# Add special tokens to vocabulary
for token in config["added_tokens"]
    content = String(token["content"])
    id = Int(token["id"])
    vocab[content] = id
end

encoder = BertTextEncoder(bpe, vocab)

# Test specific token sequences
println("\nTesting specific token sequences...")

# Test case 1
text1 = "The capital of France is [MASK]."
expected_ids1 = [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282]
tokens1 = tokenize(encoder, text1; token_ids=true)
println("\nTest case 1:")
println("Input: ", text1)
println("Expected: ", expected_ids1)
println("Got: ", tokens1)
@assert tokens1 == expected_ids1 "Token sequence 1 does not match expected output"

# Test case 2
text2 = "Hello world! This is a test."
expected_ids2 = [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282]
tokens2 = tokenize(encoder, text2; token_ids=true)
println("\nTest case 2:")
println("Input: ", text2)
println("Expected: ", expected_ids2)
println("Got: ", tokens2)
@assert tokens2 == expected_ids2 "Token sequence 2 does not match expected output"

println("\nTokenizer tests completed successfully!")
