using JSON3

# Load tokenizer configuration
vocab_path = joinpath(@__DIR__, "..", "data", "tokenizer.json")
config = JSON3.read(read(vocab_path))

# Function to get token ID from both vocab and special tokens
function get_token_id(config, token)
    # Check main vocabulary first
    id = get(config["model"]["vocab"], token, nothing)
    if id !== nothing
        return id
    end
    
    # Check special tokens
    for special in config["added_tokens"]
        if special["content"] == token
            return special["id"]
        end
    end
    return nothing
end

# Print vocabulary statistics
println("\nVocabulary Statistics:")
println("Main vocab size: ", length(config["model"]["vocab"]))
println("Special tokens: ", length(config["added_tokens"]))

# Check punctuation tokens
println("\nPunctuation token mappings:")
punctuation = [".", ",", "!", "?", ";", ":", "'", "\"", "-", "(", ")"]
for token in punctuation
    id = get_token_id(config, token)
    println("Token: '", token, "' -> ID: ", id)
end

# Check special tokens
println("\nSpecial token mappings:")
for token in config["added_tokens"]
    println("Token: '", token["content"], "' -> ID: ", token["id"])
end

# Check test case tokens
println("\nTest case token mappings:")
test_tokens = ["The", "capital", "of", "France", "is", "[MASK]", ".", "Hello", "world", "!"]
for token in test_tokens
    id = get_token_id(config, token)
    println("Token: '", token, "' -> ID: ", id)
end

# Print raw token IDs for test cases
println("\nRaw token IDs for test cases:")
println("Test case 1: The capital of France is [MASK].")
tokens1 = ["[CLS]", "The", "capital", "of", "France", "is", "[MASK]", ".", "[SEP]"]
ids1 = [get_token_id(config, t) for t in tokens1]
println("Expected: [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282]")
println("Got:      ", ids1)

println("\nTest case 2: Hello world! This is a test.")
tokens2 = ["[CLS]", "Hello", "world", "!", "This", "is", "a", "test", ".", "[SEP]"]
ids2 = [get_token_id(config, t) for t in tokens2]
println("Expected: [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282]")
println("Got:      ", ids2)
