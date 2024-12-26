using BytePairEncoding
using TextEncodeBase: Sentence

# Create GPT2 tokenizer using the built-in loader
tokenizer = BytePairEncoding.load_gpt2()

# Test cases
test_cases = [
    "Hello world",  # Basic case
    "use     multiple     spaces!",  # Multiple spaces
    "[CLS] This is a test [SEP]",  # Special tokens
    "Mr. O'Neill-McPherson's",  # Complex word
]

for (i, test) in enumerate(test_cases)
    println("\nTest case $i: '$test'")
    tokens = tokenizer(Sentence(test))
    println("Tokens:")
    token_strings = map(String, tokens)
    for (j, token) in enumerate(token_strings)
        println("  $j: '$token'")
    end
    println("Raw token count: $(length(tokens))")
    println("String token count: $(length(token_strings))")
end
