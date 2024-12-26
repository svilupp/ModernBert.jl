using JSON3
using ModernBert

# Load tokenizer
vocab_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
tokenizer = load_modernbert_tokenizer(vocab_path)

# Test words that are failing
test_words = ["capital", "France", "world", "test"]

println("Analyzing token options for problematic words:")
for word in test_words
    println("\nWord: ", word)
    
    # Check normal version
    if haskey(tokenizer.vocab, word)
        println("  Normal token ID: ", tokenizer.vocab[word])
    else
        println("  No normal token found")
    end
    
    # Check Ġ-prefixed version
    if haskey(tokenizer.vocab, "Ġ" * word)
        println("  Ġ-prefixed token ID: ", tokenizer.vocab["Ġ" * word])
    else
        println("  No Ġ-prefixed token found")
    end
end

# Print expected token IDs from test
println("\nExpected token IDs from test:")
println("capital: 5347")
println("France: 6181")
println("world: 1533")
println("test: 1071")
