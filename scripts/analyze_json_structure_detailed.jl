using JSON3

# Load vocabulary
vocab_file = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(vocab_file, String))

println("=== JSON Structure Analysis ===")
println("\nTop-level keys:")
for key in keys(config)
    println("- $key")
end

println("\nModel section keys:")
for key in keys(config.model)
    println("- $key")
end

println("\nVocabulary structure:")
vocab_section = config.model.vocab
println("Type: ", typeof(vocab_section))
println("\nFirst 10 vocabulary entries:")
entry_count = 0
for (token, id) in pairs(vocab_section)
    global entry_count += 1
    if entry_count > 10
        break
    end
    # Print token bytes for clarity
    bytes = Vector{UInt8}(token)
    println("'$token' (bytes: $bytes) -> $id")
end

# Check specific tokens from our test cases
test_tokens = [
    "The", "capital", "of", "France", "is",  # from text1
    "Hello", "world", "how", "are", "you"    # from text2
]

println("\nLooking for test tokens in vocabulary:")
for base_token in test_tokens
    # Try different variations
    variations = [
        base_token,                    # Original
        " " * base_token,             # Space prefix
        "Ġ" * base_token,             # GPT2 space prefix
        lowercase(base_token),         # Lowercase
        " " * lowercase(base_token),   # Space prefix + lowercase
        "Ġ" * lowercase(base_token)    # GPT2 space prefix + lowercase
    ]
    
    println("\nTrying variations for '$base_token':")
    for token in variations
        id = get(vocab_section, token, -1)
        bytes = Vector{UInt8}(token)
        println("  '$token' (bytes: $bytes) -> $id")
    end
end

# Print some statistics
println("\nVocabulary Statistics:")
println("Total entries: ", length(collect(pairs(vocab_section))))
println("Entries with space prefix: ", count(t -> startswith(t, " "), keys(vocab_section)))
println("Entries with Ġ prefix: ", count(t -> startswith(t, "Ġ"), keys(vocab_section)))
