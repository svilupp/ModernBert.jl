using ModernBert

# Initialize tokenizer
tokenizer = load_modernbert_tokenizer()

# Test cases focusing on edge cases that might cause infinite loops
test_cases = [
    "",            # Empty string
    " ",           # Single space
    "!",           # Single punctuation
    "Hello",       # Simple word
    "Hello!",      # Word with punctuation
]

for text in test_cases
    println("\n=== Testing: \"$text\" ===")
    @time begin
        tokens = tokenize(tokenizer, text)
        println("Generated $(length(tokens)) tokens: $tokens")
    end
end
