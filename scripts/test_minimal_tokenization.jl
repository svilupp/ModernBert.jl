using ModernBert

# Create a minimal test case with reduced complexity
function test_minimal_tokenization()
    # Initialize tokenizer with vocabulary from tokenizer.json
    tokenizer = load_modernbert_tokenizer(joinpath(@__DIR__, "..", "data", "tokenizer.json"))
    
    # Test cases focusing on core functionality
    test_cases = [
        "Hello world",  # Basic case
        "Mr. Smith"     # Period handling
    ]
    
    println("Running minimal tokenization tests...")
    for text in test_cases
        println("\nInput: ", text)
        elapsed = @elapsed tokens = tokenize(tokenizer, text)
        println("Tokens: ", tokens)
        println("Time: ", elapsed, " seconds")
    end
end

# Run tests
try
    test_minimal_tokenization()
catch e
    println("Error during tokenization: ", e)
    rethrow(e)
end
