using JSON3

function analyze_token_differences()
    config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
    config_str = read(config_path, String)
    config = JSON3.read(config_str)
    
    # Create special tokens mapping
    special_tokens = Dict{String, Int}(
        "[UNK]" => 50280,
        "[CLS]" => 50281,
        "[SEP]" => 50282,
        "[PAD]" => 50283,
        "[MASK]" => 50284
    )
    
    # Create vocabulary mapping
    vocab = Dict{String, Int}()
    
    # First, add special tokens
    for (token, id) in special_tokens
        vocab[token] = id
    end
    
    # Track skipped tokens
    skipped_tokens = Dict{String, Int}()
    
    # Then add regular vocabulary tokens
    for (token, id) in config.model.vocab
        token_str = String(token)
        if !haskey(vocab, token_str)
            vocab[token_str] = id
        else
            skipped_tokens[token_str] = id
        end
    end
    
    println("Original vocab size: ", length(config.model.vocab))
    println("Final vocab size: ", length(vocab))
    println("Special tokens: ", length(special_tokens))
    println("Skipped tokens: ", length(skipped_tokens))
    
    if !isempty(skipped_tokens)
        println("\nSkipped tokens details:")
        for (token, id) in sort(collect(skipped_tokens), by=x->x[2])
            println("Token: '", token, "' (ID: ", id, ")")
            if haskey(vocab, token)
                println("  Current ID in vocab: ", vocab[token])
            end
        end
    end
    
    # Check for gaps in token IDs
    all_ids = sort(collect(values(vocab)))
    gaps = Int[]
    for i in 1:length(all_ids)-1
        if all_ids[i+1] - all_ids[i] > 1
            for missing_id in (all_ids[i]+1):(all_ids[i+1]-1)
                push!(gaps, missing_id)
            end
        end
    end
    
    if !isempty(gaps)
        println("\nFound gaps in token IDs:")
        println(gaps)
    end
end

analyze_token_differences()
