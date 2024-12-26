using ModernBert
using Test
using BytePairEncoding
using TextEncodeBase

# Initialize tokenizer
tokenizer = ModernBertTokenizer()

# Test cases from test_tokenizer.jl
text1 = "The capital of France is [MASK]."
text2 = "Hello world! How are you?"

# Expected results
expected_ids1 = [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282]
expected_ids2 = [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282]

# Debug tokenization process
function debug_tokenization(text, expected_ids)
    println("\nDebug Tokenization for: ", text)
    println("Expected IDs: ", expected_ids)
    
    # Get actual tokens
    tokens, _, _ = encode(tokenizer, text)
    println("Actual IDs: ", tokens)
    
    # Process word by word
    words = String[]
    current_word = ""
    i = 1
    
    # Split into words while preserving punctuation
    while i <= length(text)
        if text[i] == '[' # Handle special tokens
            if !isempty(current_word)
                push!(words, current_word)
                current_word = ""
            end
            j = findnext(']', text, i)
            if j !== nothing
                push!(words, text[i:j])
                i = nextind(text, j)
                continue
            end
        elseif ispunct(text[i]) || isspace(text[i])
            if !isempty(current_word)
                push!(words, current_word)
                current_word = ""
            end
            if !isspace(text[i])
                push!(words, string(text[i]))
            end
        else
            current_word *= text[i]
        end
        i = nextind(text, i)
    end
    
    if !isempty(current_word)
        push!(words, current_word)
    end
    
    # Process each word
    for (i, word) in enumerate(words)
        println("\nWord/Token $i: '$word'")
        
        # Get tokens for this word
        word_tokens, _, _ = encode(tokenizer, word)
        println("  Word tokens: ", word_tokens)
        
        # Compare with expected tokens if this word appears in the sequence
        if !isempty(word_tokens)
            # Find this token in the expected sequence
            for token_id in word_tokens
                println("  Token ID: ", token_id)
                # Try to find where this should be in the expected sequence
                idx = findfirst(==(token_id), expected_ids)
                if idx !== nothing
                    println("    ✓ Found in expected sequence at position $idx")
                else
                    # If not found, show what we got vs what we might have expected
                    println("    ✗ Not found in expected sequence")
                    # Try to find a similar position based on word position
                    if i <= length(expected_ids)
                        println("    Expected at similar position: ", expected_ids[i])
                    end
                end
            end
        else
            println("  No tokens generated")
        end
    end
end

# Run debug analysis
println("=== Debugging Test Case 1 ===")
debug_tokenization(text1, expected_ids1)

println("\n=== Debugging Test Case 2 ===")
debug_tokenization(text2, expected_ids2)
