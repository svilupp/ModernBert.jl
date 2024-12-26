using JSON3

function analyze_vocab_ranges(vocab_path)
    vocab_data = JSON3.read(read(vocab_path, String))
    
    # Get all token IDs
    token_ids = Set{Int}()
    if haskey(vocab_data, :model) && haskey(vocab_data.model, :vocab)
        for (_, id) in pairs(vocab_data.model.vocab)
            push!(token_ids, id)
        end
    end
    
    # Find gaps in token IDs
    sorted_ids = sort(collect(token_ids))
    if !isempty(sorted_ids)
        min_id = minimum(sorted_ids)
        max_id = maximum(sorted_ids)
        println("Token ID range: $min_id to $max_id")
        
        # Find gaps
        for i in min_id:max_id
            if !(i in token_ids)
                println("Missing token ID: $i")
            end
        end
    end
    
    # Check specific ranges
    ranges = [
        (0, 255, "ASCII"),
        (256, 50279, "Base tokens"),
        (50280, 50288, "Special tokens")
    ]
    
    for (start_id, end_id, name) in ranges
        range_tokens = filter(id -> start_id <= id <= end_id, sorted_ids)
        println("\n$name range ($start_id-$end_id):")
        println("  Count: $(length(range_tokens))")
        if !isempty(range_tokens)
            println("  First: $(first(range_tokens))")
            println("  Last: $(last(range_tokens))")
        end
        
        # Find missing in this range
        missing = setdiff(Set(start_id:end_id), token_ids)
        if !isempty(missing)
            println("  Missing IDs: $missing")
        end
    end
end

vocab_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
analyze_vocab_ranges(vocab_path)
