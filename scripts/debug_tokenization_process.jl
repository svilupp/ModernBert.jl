using JSON3
using ModernBert

# Load tokenizer
vocab_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
tokenizer = load_modernbert_tokenizer(vocab_path)

# Test sentences from the test file
text1 = "The capital of France is [MASK]."
text2 = "Hello world! This is a test."

function debug_tokenization(text, tokenizer)
    println("\nTokenizing text: ", text)
    println("=" ^ 50)
    println()
    
    tokens = Int[]
    i = firstindex(text)
    last_was_space = true  # Start with true to handle first word correctly
    
    while i <= lastindex(text)
        # Skip multiple spaces
        while i <= lastindex(text) && isspace(text[i])
            println("Skipping space at position ", i)
            i = nextind(text, i)
            last_was_space = true
            continue
        end
        
        if i > lastindex(text)
            break
        end
        
        # Try to find the longest matching token at current position
        longest_match = ""
        longest_id = nothing
        current_idx = i
        current_text = ""
        
        println("\nTrying to find longest match starting at position ", i, " (after space: ", last_was_space, ")")
        
        # First check for special tokens
        for (token, id) in tokenizer.special_tokens
            if startswith(text[i:end], token)
                println("  Checking special token: ", token)
                if length(token) > length(longest_match)
                    println("    Found longer special token match: ", token, " => ", id)
                    longest_match = token
                    longest_id = id
                end
            end
        end
        
        # If no special token found, try regular tokens
        if isempty(longest_match)
            while current_idx <= lastindex(text)
                # Stop at space
                if isspace(text[current_idx])
                    break
                end
                
                current_text *= text[current_idx]
                println("\n  Trying text: ", current_text)
                
                # Try both normal and Ġ-prefixed versions
                if haskey(tokenizer.vocab, current_text)
                    println("    Found normal token: ", current_text, " => ", tokenizer.vocab[current_text])
                end
                
                if haskey(tokenizer.vocab, "Ġ" * current_text)
                    println("    Found Ġ-prefixed token: Ġ", current_text, " => ", tokenizer.vocab["Ġ" * current_text])
                end
                
                # For punctuation and special characters, prefer non-prefixed
                if haskey(tokenizer.vocab, current_text) && 
                   length(current_text) == 1 && 
                   (ispunct(current_text[1]) || current_text[1] in ['[', ']', '.', ',', '!', '?', '-', '@', '{', '}'])
                    println("    Selected punctuation token: ", current_text, " => ", tokenizer.vocab[current_text])
                    longest_match = current_text
                    longest_id = tokenizer.vocab[current_text]
                end
                
                # For words after space, prefer Ġ-prefixed version
                if last_was_space && haskey(tokenizer.vocab, "Ġ" * current_text)
                    println("    Selected Ġ-prefixed token after space: Ġ", current_text, " => ", tokenizer.vocab["Ġ" * current_text])
                    longest_match = current_text
                    longest_id = tokenizer.vocab["Ġ" * current_text]
                # For words not after space, try non-prefixed first
                elseif haskey(tokenizer.vocab, current_text) && (length(current_text) > length(longest_match))
                    println("    Selected longer normal token: ", current_text, " => ", tokenizer.vocab[current_text])
                    longest_match = current_text
                    longest_id = tokenizer.vocab[current_text]
                end
                
                current_idx = nextind(text, current_idx)
            end
        end
        
        # If we found a match, add it and advance
        if !isempty(longest_match)
            push!(tokens, longest_id)
            println("\n  Final selection: ", longest_match, " => ", longest_id)
            for _ in 1:length(longest_match)
                i = nextind(text, i)
            end
        else
            # No match found, emit [UNK] and advance one character
            push!(tokens, tokenizer.special_tokens["[UNK]"])
            println("\n  No match found, using [UNK] token")
            i = nextind(text, i)
        end
        
        last_was_space = false
    end
    
    println("\nFinal tokens: ", tokens)
    return tokens
end

println("\n=== Testing first sentence ===")
tokens1 = debug_tokenization(text1, tokenizer)
println("\nExpected tokens: [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282]")
println("Got tokens:      ", tokens1)

println("\n=== Testing second sentence ===")
tokens2 = debug_tokenization(text2, tokenizer)
println("\nExpected tokens: [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282]")
println("Got tokens:      ", tokens2)
