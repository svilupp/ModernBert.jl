using ModernBert

# Initialize tokenizer
vocab_path = joinpath(@__DIR__, "..", "data", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"
tokenizer = load_modernbert_tokenizer(vocab_path)

# Test special token handling
function test_special_tokens()
    # Test [MASK] token specifically
    text = "The capital of France is [MASK]."
    println("\nTesting: ", text)
    tokens = tokenize(tokenizer, text)
    println("Tokens: ", tokens)
    
    # Test word boundary after punctuation
    text2 = "Mr. O'Neill"
    println("\nTesting: ", text2)
    tokens2 = tokenize(tokenizer, text2)
    println("Tokens: ", tokens2)
end

# Run test
test_special_tokens()
