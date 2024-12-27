# Add project directory to load path
push!(LOAD_PATH, dirname(dirname(@__FILE__)))

using ModernBert

# Initialize tokenizer with minimal vocabulary
tokenizer = load_modernbert_tokenizer()

# Define minimal set of test cases focusing on key functionality
test_cases = [
    "Hello world!",  # Basic ASCII text
    "H₂O",          # Subscript handling
    "~7.0",         # Special characters
    "你好",         # UTF-8 characters
    "[MASK]",       # Special token
]

println("Running minimal test cases...")
for (i, text) in enumerate(test_cases)
    print("Test case $i: ")
    @time tokens = tokenize(tokenizer, text)
    println("Input: $text")
    println("Tokens: $tokens")
    println("Token IDs: ", join(tokens, ", "))
    println()
end

# Verify special token handling
special_tokens = ["[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"]
println("\nVerifying special tokens...")
for token in special_tokens
    @time tokens = tokenize(tokenizer, token)
    println("Token: $token")
    println("Token IDs: ", join(tokens, ", "))
end
