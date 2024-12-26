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

# Print first few entries directly
println("\nFirst 10 vocabulary entries (raw):")
entry_count = 0
for pair in pairs(vocab_section)
    global entry_count += 1
    if entry_count > 10
        break
    end
    token, id = pair
    println("'$token' -> $id")
end

# Check special tokens
special_tokens = ["[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"]
println("\nSpecial Token Analysis:")
for token in special_tokens
    id = get(vocab_section, token, -1)
    println("'$token' -> $id")
end

# Check test case tokens
test_tokens = [
    "The", "capital", "of", "France", "is",  # from text1
    "Hello", "world", "how", "are", "you"    # from text2
]

println("\nTest Token Analysis:")
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
    
    println("\nVariations for '$base_token':")
    for token in variations
        id = get(vocab_section, token, -1)
        println("  '$token' -> $id")
    end
end

# Print some statistics
println("\nVocabulary Statistics:")
println("Total entries: ", length(collect(pairs(vocab_section))))
println("Entries with space prefix: ", count(t -> startswith(t, " "), keys(vocab_section)))
println("Entries with Ġ prefix: ", count(t -> startswith(t, "Ġ"), keys(vocab_section)))

# Sample some entries with Ġ prefix
println("\nSample entries with Ġ prefix:")
count = 0
for (token, id) in pairs(vocab_section)
    if startswith(token, "Ġ")
        println("'$token' -> $id")
        count += 1
        count >= 5 && break
    end
end
