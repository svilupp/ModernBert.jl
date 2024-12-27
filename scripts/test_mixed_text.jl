using ModernBert

# Initialize tokenizer
vocab_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"
tokenizer = load_modernbert_tokenizer(vocab_path)

# Test case that was failing
text = "Mr. O'Neill-McPherson's co-workers @ ABC.com [and] {Dr. J.R.R. Martin-Smith} use     multiple     spaces!"
println("Testing text: ", text)

# Time the tokenization and encoding
tokens, token_types, attention_mask = @time encode(tokenizer, text)

# Print token IDs and their corresponding tokens
println("\nTokens (with token types and attention mask):")
for (i, (token_id, type_id, mask)) in enumerate(zip(tokens, token_types, attention_mask))
    token = get(tokenizer.id_to_token, token_id, "<unknown>")
    println("$i: $token_id ($token) [type=$type_id, mask=$mask]")
end

# Expected token sequence for comparison
expected_ids = [50281, 7710, 15, 473, 8, 41437, 14, 11773, 49, 379,
    1665, 434, 820, 14, 26719, 1214, 15599, 15, 681, 544,
    395, 62, 551, 9034, 15, 500, 15, 51, 15, 51,
    15, 8698, 14, 21484, 94, 897, 50273, 34263, 50273, 31748,
    2, 50282]

println("\nComparing with expected sequence:")
for (i, (actual, expected)) in enumerate(zip(tokens, expected_ids))
    actual_token = get(tokenizer.id_to_token, actual, "<unknown>")
    expected_token = get(tokenizer.id_to_token, expected, "<unknown>")
    if actual != expected
        println("Mismatch at position $i: got $actual ($actual_token), expected $expected ($expected_token)")
    end
end
