using ModernBert
using ModernBert.ModernBertTokenizerImpl: KNOWN_TOKENS, REQUIRED_TOKENS, SPECIAL_TOKENS, load_modernbert_tokenizer, tokenize

# Initialize tokenizer
tokenizer = load_modernbert_tokenizer()

# Print vocabulary loading order
println("Checking vocabulary loading order...")
println("KNOWN_TOKENS:")
for (token, id) in KNOWN_TOKENS
    println("  $token => $id")
end

println("\nSPECIAL_TOKENS:")
for (token, id) in SPECIAL_TOKENS
    println("  $token => $id")
end

println("\nREQUIRED_TOKENS:")
for (token, id) in REQUIRED_TOKENS
    println("  $token => $id")
end

# Test words and their expected IDs
test_words = [
    "capital" => 5347,
    "of" => 273,
    "France" => 6181,
    "is" => 310
]

println("Checking token IDs in vocabulary...")
for (word, expected_id) in test_words
    actual_id = nothing
    
    # Check in KNOWN_TOKENS
    if haskey(KNOWN_TOKENS, word)
        actual_id = KNOWN_TOKENS[word]
        println("$word found in KNOWN_TOKENS with ID: $actual_id (expected: $expected_id)")
    end
    
    # Check in vocab
    if haskey(tokenizer.vocab, word)
        vocab_id = tokenizer.vocab[word]
        if actual_id !== nothing && vocab_id != actual_id
            println("WARNING: $word has different ID in vocab: $vocab_id vs KNOWN_TOKENS: $actual_id")
        else
            actual_id = vocab_id
            println("$word found in vocab with ID: $actual_id (expected: $expected_id)")
        end
    end
    
    # Check Ġ-prefixed version
    prefixed = "Ġ" * word
    if haskey(KNOWN_TOKENS, prefixed)
        prefixed_id = KNOWN_TOKENS[prefixed]
        println("$prefixed found in KNOWN_TOKENS with ID: $prefixed_id")
    end
    if haskey(tokenizer.vocab, prefixed)
        prefixed_vocab_id = tokenizer.vocab[prefixed]
        println("$prefixed found in vocab with ID: $prefixed_vocab_id")
    end
    
    println()
end

# Test actual tokenization
test_texts = [
    "The capital of France is [MASK].",
    "Hello world! This is a test."
]

println("\nTesting tokenization...")
for text in test_texts
    tokens = tokenize(tokenizer, text)
    println("Text: $text")
    println("Tokens: $tokens")
    println()
end
