using ModernBert

# Load tokenizer
vocab_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
tokenizer = load_modernbert_tokenizer(vocab_path)

# Test cases (reduced for CI performance)
test_cases = [
    "Hello world",  # Basic case
    "The quick brown fox jumps over the lazy dog.",  # Common sentence
    "xyz123",  # Unknown token
]

# Benchmark each case
println("Benchmarking tokenization performance:")
println("=====================================")

for (i, text) in enumerate(test_cases)
    println("\nCase $i: ", text)
    print("Tokenization time: ")
    @time tokens = tokenize(tokenizer, text)
    println("Tokens: ", tokens)
    println("Token count: ", length(tokens))
    
    print("Full encoding time: ")
    @time tokens, types, mask = encode(tokenizer, text)
    println("Encoded tokens: ", tokens)
    println("Token count: ", length(tokens))
end

# Test batch processing performance
texts = repeat(test_cases, 10)  # Create a larger batch
print("\nBatch processing time ($(length(texts)) texts): ")
@time results = encode(tokenizer, texts)
println("Total tokens: ", sum(results[3]))
