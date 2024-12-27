using ModernBert
using ModernBert.ModernBertTokenizerImpl

# Initialize tokenizer
vocab_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
tokenizer = load_modernbert_tokenizer(vocab_path)

function debug_tokenization(text)
    println("\nDebug Tokenization for: ", text)
    println("=" ^ 50)
    
    # Track current position
    current_idx = firstindex(text)
    while current_idx <= lastindex(text)
        # Get the next token
        token, id = find_longest_token(tokenizer, text, current_idx)
        
        # Print debug info
        println("Position: ", current_idx)
        println("Current text: ", text[current_idx:end])
        println("Found token: ", token)
        println("Token ID: ", id)
        println("-" ^ 30)
        
        # Advance position
        current_idx = current_idx + length(token)
    end
    
    # Show final tokenization
    tokens, _, _ = encode(tokenizer, text)
    println("\nFinal Tokenization:")
    println("Tokens: ", [get(tokenizer.id_to_token, t, string(t)) for t in tokens])
    println("Token IDs: ", tokens)
end

# Test cases
debug_tokenization("Mr.")
debug_tokenization("Mr. O'Neill-McPherson's")
