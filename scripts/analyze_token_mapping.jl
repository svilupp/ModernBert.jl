using JSON3
using ModernBert

# Load tokenizer configuration
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path))

# Initialize tokenizer
tokenizer = ModernBertTokenizer(config_path)

# Test cases from failing tests
text1 = "The capital of France is [MASK]."
text2 = "Python is a great programming language."

# Print vocabulary entries for key words
println("Vocabulary mappings for key words:")
for word in ["The", "capital", "France", "Python", "great", "programming"]
    id = get(tokenizer.vocab, word, nothing)
    println("'$word' => $id")
end

# Print actual tokenization
println("\nTokenization for text1:")
tokens1, _, _ = encode(tokenizer, text1)
println(tokens1)

println("\nTokenization for text2:")
tokens2, _, _ = encode(tokenizer, text2)
println(tokens2)

# Print expected vs actual for comparison
expected_ids1 = [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282]
expected_ids2 = [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282]

println("\nComparison for text1:")
for (exp, act) in zip(expected_ids1, tokens1)
    println("Expected: $exp, Got: $act")
end

println("\nComparison for text2:")
for (exp, act) in zip(expected_ids2, tokens2)
    println("Expected: $exp, Got: $act")
end
