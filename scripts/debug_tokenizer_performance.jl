using ModernBert

# Create test cases that might trigger slow performance
function create_test_cases()
    return [
        # Basic cases
        "Hello world",
        # Long text with repeated patterns
        repeat("test " * join('a':'z'), 100),
        # Text with lots of punctuation
        repeat("...!!!???---@@@", 100),
        # Mixed case with spaces
        repeat("A B C D E F G", 100),
        # Unicode text
        repeat("Hello 世界", 100),
        # Edge cases that might trigger slow matching
        repeat("a" * repeat(".", 100), 10),  # Long punctuation sequences
        repeat(" " * repeat("a", 50), 10),   # Long word after space
        join(repeat(['[', ']'], 1000)),      # Many special characters
        repeat("Ġ", 1000),                   # Many Ġ prefixes
        repeat("test", 1000)                 # Simple repeated word
    ]
end

# Initialize tokenizer
tokenizer = ModernBertTokenizer()

# Test each case and measure time
println("Running performance tests...")
for (i, test_case) in enumerate(create_test_cases())
    println("\nTest case $i:")
    println("Length: $(length(test_case)) characters")
    println("Preview: $(test_case[1:min(50, end)])...")
    
    # Warm up run
    tokenize(tokenizer, test_case)
    
    # Timed run
    elapsed = @elapsed tokens = tokenize(tokenizer, test_case)
    
    println("Time: $(elapsed) seconds")
    println("Tokens generated: $(length(tokens))")
    
    # Early warning if taking too long
    if elapsed > 1.0
        println("WARNING: Test case $i is taking >1 second to process!")
    end
    
    # If taking too long, try to identify where it's slow
    if elapsed > 5.0
        println("CRITICAL: Test case $i is extremely slow!")
        println("Full test case:")
        println(test_case)
    end
end
