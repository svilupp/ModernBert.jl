using ModernBert

# Initialize tokenizer
vocab_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"
tokenizer = load_modernbert_tokenizer(vocab_path)

# Test case
text = "Mr. O'Neill-McPherson's"
tokens = tokenize(tokenizer, text)

# Print token IDs and their string representations
println("Token sequence for: ", text)
for token_id in tokens
    token_str = get(tokenizer.id_to_token, token_id, "<unknown>")
    println("ID: $token_id -> Token: $token_str")
end

# Expected sequence
expected_ids = [7710, 15, 473, 8, 41437, 14, 11773]
expected_tokens = [get(tokenizer.id_to_token, id, "<unknown>") for id in expected_ids]

println("\nExpected sequence:")
for (id, token) in zip(expected_ids, expected_tokens)
    println("ID: $id -> Token: $token")
end

# Verify specific tokens
mr_token = "Mr"
o_token = "ĠO"
println("\nVerifying specific tokens:")
println("Mr token ID: ", get(tokenizer.known_tokens, mr_token, get(tokenizer.vocab, mr_token, -1)))
println("ĠO token ID: ", get(tokenizer.known_tokens, o_token, get(tokenizer.vocab, o_token, -1)))
