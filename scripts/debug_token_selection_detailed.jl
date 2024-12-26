using JSON3
using ModernBert

# Load vocabulary
vocab_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
vocab = JSON3.read(read(vocab_path, String))

# Test text
text = "The capital of France is [MASK]."

# Function to check token variants
function check_token_variants(word)
    println("\nChecking variants for word: '$word'")
    
    # Check normal variant
    normal_id = get(vocab.model.vocab, Symbol(word), nothing)
    println("  Normal variant ('$word'): ", normal_id)
    
    # Check Ġ-prefixed variant
    g_word = "Ġ" * word
    g_id = get(vocab.model.vocab, Symbol(g_word), nothing)
    println("  Ġ-prefixed variant ('$g_word'): ", g_id)
end

# Split text into words and check each
words = split(text)
for word in words
    check_token_variants(word)
end

# Load tokenizer and show actual tokenization
tokenizer = load_modernbert_tokenizer(vocab_path)
tokens, _, _ = encode(tokenizer, text)
println("\nFull tokenization result:")
println("Text: ", text)
println("Token IDs: ", tokens)

# Print expected vs actual for each word
expected_ids = Dict(
    "The" => 510,
    "capital" => 38479,
    "of" => 1171,
    "France" => 33639,
    "is" => 261,
    "[MASK]" => 50284,
    "." => 15
)

println("\nToken comparison:")
for (word, expected) in expected_ids
    actual = nothing
    if word in words
        actual = tokens[findfirst(==(word), words) + 1]  # +1 for [CLS]
    end
    println("Word: '$word'")
    println("  Expected ID: ", expected)
    println("  Actual ID: ", actual)
    println("  Matches: ", expected == actual)
end
