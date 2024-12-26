using JSON3

function analyze_vocabulary()
    # Load config
    config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
    config = JSON3.read(read(config_path, String))

    # Special token IDs
    special_tokens = Dict{String, Int}(
        "[UNK]" => 50280,
        "[CLS]" => 50281,
        "[SEP]" => 50282,
        "[PAD]" => 50283,
        "[MASK]" => 50284
    )

    # Initialize vocabulary
    vocab = Dict{String, Int}()
    
    # Add special tokens first
    for (token, id) in special_tokens
        vocab[token] = id
    end
    
    # Track token counts and conflicts
    regular_tokens = 0
    special_token_conflicts = 0
    special_token_ids = Set(values(special_tokens))
    
    # Add regular vocabulary and track statistics
    skipped_tokens = Dict{String, Int}()
    for (token, id) in pairs(config.model.vocab)
        token_str = String(token)
        if !haskey(special_tokens, token_str)
            vocab[token_str] = id
            regular_tokens += 1
            if id in special_token_ids
                special_token_conflicts += 1
                println("Token conflict: $token_str (ID: $id)")
            end
        else
            skipped_tokens[token_str] = id
        end
    end
    
    # Print statistics
    println("\nVocabulary Statistics:")
    println("Total vocabulary size: ", length(vocab))
    println("Regular tokens: ", regular_tokens)
    println("Special tokens: ", length(special_tokens))
    println("Special token ID conflicts: ", special_token_conflicts)
    println("\nSkipped Tokens:")
    for (token, id) in sort(collect(skipped_tokens))
        println("$token => $id")
    end
    
    # Find highest token ID
    max_id = maximum(values(vocab))
    println("\nHighest token ID: ", max_id)
    
    # Check for gaps in token IDs
    println("\nGaps in token IDs:")
    all_ids = Set(values(vocab))
    for i in 0:max_id
        if i ∉ all_ids && i < 50280  # Exclude special token range
            println("Missing token ID: ", i)
        end
    end
    
    return vocab
end

# Run analysis
vocab = analyze_vocabulary()
