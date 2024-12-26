using JSON3

function analyze_vocab()
    config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
    config_str = read(config_path, String)
    config = JSON3.read(config_str)
    
    # Count total vocabulary size
    vocab_size = length(config.model.vocab)
    println("Vocabulary size: ", vocab_size)
    
    # Check special tokens
    special_tokens = Dict(
        "[CLS]" => 50281,
        "[SEP]" => 50282,
        "[MASK]" => 50284,
        "[PAD]" => 50283,
        "[UNK]" => 50280
    )
    
    println("\nChecking special tokens:")
    for (token, expected_id) in special_tokens
        if haskey(config.model.vocab, token)
            actual_id = config.model.vocab[token]
            println("$token: expected=$expected_id, actual=$actual_id")
        else
            println("$token: not found in vocab")
        end
    end
    
    # Find highest token ID
    max_id = 0
    for (_, id) in config.model.vocab
        max_id = max(max_id, id)
    end
    println("\nHighest token ID: ", max_id)
end

analyze_vocab()
