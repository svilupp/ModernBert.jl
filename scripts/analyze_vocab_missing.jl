using JSON3

# Load config
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path, String))

# Count total vocabulary size
vocab_size = length(config.model.vocab)
println("Total vocabulary size in config: ", vocab_size)

# Special tokens from our implementation
special_tokens = Dict(
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[MASK]" => 50284,
    "[PAD]" => 50283,
    "[UNK]" => 50280
)

# Check if special tokens exist in config vocabulary
for (token, id) in special_tokens
    if haskey(config.model.vocab, token)
        println("Special token $token found in vocab with ID: ", config.model.vocab[token])
    else
        println("Special token $token NOT found in vocab")
    end
end

# Print tokens with IDs around special token range
println("\nTokens with IDs around special token range (50275-50290):")
for (token, id) in pairs(config.model.vocab)
    if 50275 <= id <= 50290
        println("$token => $id")
    end
end
