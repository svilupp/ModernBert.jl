using ModernBert

# Load tokenizer
vocab_path = joinpath(dirname(@__DIR__), "test", "model", "tokenizer.json")
tokenizer = load_modernbert_tokenizer(vocab_path)

# Test case with mixed text (reduced from full test)
test_cases = [
    "Hello world! This is a test.", # Basic ASCII for baseline
    "The pH of H₂O is ~7.0", # Subscripts and special chars
    "你好" # Minimal UTF-8 test
]

# Expected token IDs for validation
expected_ids = [
    [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282],  # Hello world! This is a test.
    [50281, 510, 8487, 273, 388, 46979, 213, 48, 310, 5062, 24, 15, 17, 50282],  # The pH of H₂O is ~7.0
    [50281, 50280, 50282]  # 你好 (unknown tokens)
]

# Clear tokenizer cache
empty!(tokenizer.cache)

# Process each test case individually for debugging
println("\nTesting individual cases:")
for (i, text) in enumerate(test_cases)
    tokens, _, _ = encode(tokenizer, text)
    println("\nCase $i: ", text)
    println("Expected: ", expected_ids[i])
    println("Got:      ", tokens)
    
    # Compare tokens one by one
    if length(tokens) != length(expected_ids[i])
        println("Length mismatch: got $(length(tokens)), expected $(length(expected_ids[i]))")
    else
        for (j, (got, expected)) in enumerate(zip(tokens, expected_ids[i]))
            if got != expected
                println("Mismatch at position $j:")
                println("  Expected: $expected ($(get(tokenizer.id_to_token, expected, "UNKNOWN")))")
                println("  Got:      $got ($(get(tokenizer.id_to_token, got, "UNKNOWN")))")
            end
        end
    end
end

# Test batch encoding
println("\nTesting batch encoding:")
@time begin
    tokens = encode(tokenizer, test_cases)
    println("Batch encoding result shape: ", size(tokens[1]))
end

# Compare batch results with individual results
for j in 1:length(test_cases)
    individual_tokens, _, _ = encode(tokenizer, test_cases[j])
    batch_tokens = tokens[1][:, j]
    
    # Trim padding tokens for comparison
    while !isempty(batch_tokens) && batch_tokens[end] == 50283  # [PAD] token
        pop!(batch_tokens)
    end
    
    if individual_tokens != batch_tokens
        println("\nMismatch in batch vs individual encoding for case $j:")
        println("Individual: ", individual_tokens)
        println("Batch:      ", batch_tokens)
    end
end
