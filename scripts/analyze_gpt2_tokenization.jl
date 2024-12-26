using BytePairEncoding
using TextEncodeBase

# Initialize GPT2 tokenizer
tkr = BytePairEncoding.load_gpt2()

# Test string with spaces
test_str = "Mr. O'Neill-McPherson's co-workers"

# Analyze raw tokenization
tokens = tkr(TextEncodeBase.Sentence(test_str))
println("Raw tokens for: ", test_str)
for (i, token) in enumerate(tokens)
    # Print token and its byte representation
    bytes = Vector{UInt8}(token)
    println("Token $i: '$(token)' (bytes: $(bytes))")
end

# Test with explicit spaces
test_str_spaces = "   multiple   spaces   "
tokens_spaces = tkr(TextEncodeBase.Sentence(test_str_spaces))
println("\nRaw tokens for spaces: ", test_str_spaces)
for (i, token) in enumerate(tokens_spaces)
    bytes = Vector{UInt8}(token)
    println("Token $i: '$(token)' (bytes: $(bytes))")
end
