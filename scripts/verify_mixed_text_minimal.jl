using ModernBert

# Initialize tokenizer
tokenizer = load_modernbert_tokenizer()

# Test with minimal set of mixed text cases that cover different scenarios
test_cases = [
    "Hello world! This is a test.",  # Basic case
    "The pH of H₂O is ~7.0",         # Special characters
    "CEO@company.com met w/ VP",      # Email and abbreviations
]

println("\n=== Running Minimal Mixed Text Tests ===")
for (i, text) in enumerate(test_cases)
    println("\nTest Case $i: \"$text\"")
    elapsed = @elapsed begin
        tokens, types, mask = encode(tokenizer, text)
    end
    println("Tokens: ", tokens)
    println("Time: $(round(elapsed, digits=6)) seconds")
end

# Test batch processing
println("\n=== Testing Batch Processing ===")
elapsed_batch = @elapsed begin
    tokens, types, mask = encode(tokenizer, test_cases)
end
println("Batch processing time: $(round(elapsed_batch, digits=6)) seconds")
println("Token matrix size: ", size(tokens))
