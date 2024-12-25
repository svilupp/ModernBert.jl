using JSON3
include("../src/bpe.jl")

# Load tokenizer configuration
vocab_path = joinpath(@__DIR__, "..", "data", "tokenizer.json")
config = JSON3.read(read(vocab_path))

# Create tokenizer
tokenizer = load_tokenizer(vocab_path)

# Debug function to show BPE merge steps
function debug_bpe_merge(tokenizer::BPETokenizer, word::String)
    println("\nDebugging BPE merge for word: '$word'")
    
    # Add 'Ġ' prefix for word boundaries
    word_with_prefix = "Ġ" * word
    chars = string.(collect(word_with_prefix))
    println("Initial chars: ", chars)
    
    iteration = 1
    while true
        pairs = get_pairs(chars)
        isempty(pairs) && break
        
        println("\nIteration $iteration:")
        println("Current tokens: ", chars)
        println("Available pairs: ", pairs)
        
        # Find the highest priority pair
        best_pair = nothing
        best_pair_idx = nothing
        for (i, pair) in enumerate(tokenizer.merges)
            if pair in pairs
                best_pair = pair
                best_pair_idx = i
                break
            end
        end
        
        if isnothing(best_pair)
            println("No valid merge found, stopping")
            break
        end
        
        println("Best pair (priority $best_pair_idx): ", best_pair)
        
        # Merge the pair throughout the word
        new_chars = Vector{String}()
        i = 1
        while i <= length(chars)
            if i < length(chars) && chars[i] == best_pair[1] && chars[i+1] == best_pair[2]
                merged = chars[i] * chars[i+1]
                push!(new_chars, merged)
                println("Merging at position $i: $(chars[i]) + $(chars[i+1]) -> $merged")
                i += 2
            else
                push!(new_chars, chars[i])
                i += 1
            end
        end
        chars = new_chars
        iteration += 1
    end
    
    println("\nFinal tokens: ", chars)
    token_ids = [get(tokenizer.vocab, t, tokenizer.special_tokens["[UNK]"]) for t in chars]
    println("Token IDs: ", token_ids)
    return chars, token_ids
end

# Test problematic words from our test cases
println("\n=== Testing 'capital' ===")
tokens, ids = debug_bpe_merge(tokenizer, "capital")
println("Expected ID: 5347, Got: ", ids)

println("\n=== Testing 'France' ===")
tokens, ids = debug_bpe_merge(tokenizer, "France")
println("Expected ID: 6181, Got: ", ids)

# Print first 20 merge rules for inspection
println("\n=== First 20 merge rules ===")
for (i, rule) in enumerate(first(tokenizer.merges, 20))
    println("$i: $rule")
end

# Print vocabulary entries for our test words
println("\n=== Vocabulary entries for test words ===")
for prefix in ["Ġ", ""]
    for word in ["capital", "france", "France", "the", "of", "is"]
        token = prefix * word
        id = get(tokenizer.vocab, token, nothing)
        if !isnothing(id)
            println("'$token' -> $id")
        end
    end
end
