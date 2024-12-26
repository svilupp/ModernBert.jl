using JSON3

function analyze_vocab()
    config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
    config = JSON3.read(read(config_path))
    
    # Count total vocabulary size
    vocab_size = length(config.model.vocab)
    println("Vocabulary size: ", vocab_size)
    
    # Check special tokens
    special_tokens = Dict{String,Int}(
        "[CLS]" => 50281,
        "[SEP]" => 50282,
        "[MASK]" => 50284,
        "[PAD]" => 50283,
        "[UNK]" => 50280
    )
    
    for (token, id) in special_tokens
        if haskey(config.model.vocab, token)
            println("Special token $token found with ID: ", config.model.vocab[token])
        else
            println("Special token $token not found in vocabulary")
        end
    end
    
    # Print first few regular tokens
    println("\nFirst 10 regular tokens:")
    count = 0
    for (token, id) in config.model.vocab
        if !haskey(special_tokens, token)
            println("$token => $id")
            count += 1
            if count >= 10
                break
            end
        end
    end
end

analyze_vocab()
