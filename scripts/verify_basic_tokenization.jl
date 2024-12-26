using BytePairEncoding
using TextEncodeBase
using JSON3
using ModernBert

# Load test configuration
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path))

# Initialize tokenizer
tokenizer = ModernBertTokenizer(config_path)

# Test 1: Verify special token IDs
println("Special Token ID Verification:")
special_tokens = Dict(
    "[UNK]" => 50280,
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[PAD]" => 50283,
    "[MASK]" => 50284
)

for (token, expected_id) in special_tokens
    actual_id = tokenizer.special_tokens[token]
    match = actual_id == expected_id
    println("$token: Expected=$expected_id, Got=$actual_id, Match=$(match ? "✓" : "✗")")
end

# Test 2: Verify basic vocabulary tokens
println("\nBasic Vocabulary Token Verification:")
test_tokens = ["The", "capital", "France", "Python"]
for token in test_tokens
    id = get(tokenizer.vocab, token, nothing)
    println("'$token' => $id")
end

# Test 3: Basic tokenization test
println("\nBasic Tokenization Test:")
test_sentences = [
    "The capital of France is [MASK].",
    "Python is a great programming language."
]

expected_ids = Dict(
    "The capital of France is [MASK]." => [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282],
    "Python is a great programming language." => [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282]
)

for sentence in test_sentences
    println("\nSentence: $sentence")
    tokens, _, _ = encode(tokenizer, sentence)
    expected = expected_ids[sentence]
    
    println("Expected: $expected")
    println("Got:      $tokens")
    
    # Compare token by token
    println("\nToken comparison:")
    for (i, (exp, got)) in enumerate(zip(expected, tokens))
        exp_token = get(tokenizer.id_to_token, exp, "UNKNOWN")
        got_token = get(tokenizer.id_to_token, got, "UNKNOWN")
        match = exp == got
        println("Position $i: Expected $exp ($exp_token) -> Got $got ($got_token) $(match ? "✓" : "✗")")
    end
end
