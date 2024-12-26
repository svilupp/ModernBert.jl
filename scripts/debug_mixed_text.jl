using ModernBert

# Initialize tokenizer
vocab_path = joinpath(dirname(@__DIR__), "test", "model", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"
tokenizer = load_modernbert_tokenizer(vocab_path)

# Test cases that cover different scenarios
test_cases = [
    "Mr. O'Neill-McPherson's co-workers @ ABC.com",  # Basic case with special chars
    "C++ programmers write \"Hello, World!\"",       # Programming syntax
    "The pH of H₂O is ~7.0",                        # Scientific notation
    "你好,世界!Hello,世界!"                           # Mixed UTF-8
]

println("=== Testing Individual Cases ===")
for (i, text) in enumerate(test_cases)
    println("\nCase $i: ", text)
    time_result = @timed encode(tokenizer, text)
    tokens, types, mask = time_result.value
    println("Time: $(time_result.time * 1000)ms")
    println("Token IDs: ", tokens)
    println("Token values: ", [get(tokenizer.id_to_token, t, "<UNK>") for t in tokens])
end

println("\n=== Testing Batch Processing ===")
time_result = @timed encode(tokenizer, test_cases)
tokens, types, mask = time_result.value
println("Batch processing time: $(time_result.time * 1000)ms")
println("Token matrix shape: ", size(tokens))
for j in 1:size(tokens, 2)
    println("\nSequence $j tokens: ", tokens[:, j])
    println("Original text: ", test_cases[j])
end
