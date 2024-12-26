using JSON3

# Load vocabulary file
vocab_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found at $(vocab_path)"

vocab_data = JSON3.read(read(vocab_path, String))

# Test tokens we need to verify
test_tokens = [
    "The",      # Expected: 510
    "capital",  # Expected: 5347
    "of",       # Expected: 273
    "France",   # Expected: 6181
    "is",       # Expected: 310
    ".",        # Expected: 15
]

# Check both with and without Ġ prefix
for token in test_tokens
    # Check direct token
    if haskey(vocab_data.model.vocab, Symbol(token))
        println("Token: $(token), ID: $(vocab_data.model.vocab[Symbol(token)])")
    else
        println("Token: $(token) not found directly")
    end
    
    # Check with Ġ prefix
    if haskey(vocab_data.model.vocab, Symbol("Ġ" * token))
        println("Token: Ġ$(token), ID: $(vocab_data.model.vocab[Symbol("Ġ" * token)])")
    else
        println("Token: Ġ$(token) not found")
    end
end

# Print special tokens section if it exists
if haskey(vocab_data, :added_tokens)
    println("\nSpecial Tokens:")
    for token in vocab_data.added_tokens
        println("$(token.content): $(token.id)")
    end
end
