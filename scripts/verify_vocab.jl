using JSON3

function verify_vocab()
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
    
    # Then add regular vocabulary tokens
    for (token, id) in config.model.vocab
        token_str = String(token)
        if !haskey(vocab, token_str)
            vocab[token_str] = id
        end
    end
    
    println("Total vocabulary size: ", length(vocab))
    println("\nSpecial tokens in vocabulary:")
    for (token, expected_id) in special_tokens
        actual_id = get(vocab, token, -1)
        println("$token: expected=$expected_id, actual=$actual_id")
    end
    
    # Find highest and lowest token IDs
    max_id = maximum(values(vocab))
    min_id = minimum(values(vocab))
    println("\nToken ID range:")
    println("Minimum ID: ", min_id)
    println("Maximum ID: ", max_id)
end

verify_vocab()
