using JSON3

# Load vocabulary
vocab_file = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(vocab_file, String))
vocab = Dict(token => id for (token, id) in pairs(config.model.vocab))

# Test strings from our failing tests
test_tokens = [
    "The", "Ġcapital", "Ġof", "ĠFrance", "Ġis",  # from text1
    "Hello", "Ġworld", ",", "Ġhow", "Ġare", "Ġyou", "?"  # from text2
]

# Expected IDs from test
expected_ids1 = [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282]
expected_ids2 = [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282]

println("=== Vocabulary Analysis ===")
println("\nVocabulary size: ", length(vocab))

# Check for space-prefixed versions
println("\n=== Token Mapping Analysis ===")
for token in test_tokens
    # Try different variations
    variations = [
        token,                    # Original
        replace(token, "Ġ" => " "),  # Replace Ġ with space
        replace(token, "Ġ" => ""),   # Remove Ġ
    ]
    
    println("\nToken: '$token'")
    for var in variations
        id = get(vocab, var, -1)
        println("  Variation: '$var' -> ID: $id")
    end
end

# Print some sample vocabulary entries
println("\n=== Sample Vocabulary Entries ===")
sample_size = 10
sample_tokens = collect(keys(vocab))[1:sample_size]
for token in sample_tokens
    println("'$token' -> $(vocab[token])")
end
