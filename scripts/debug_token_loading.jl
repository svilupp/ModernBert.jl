using JSON3

function debug_token_loading(vocab_path)
    println("=== Starting Token Loading Debug ===")
    
    # Initialize special tokens
    special_tokens = Dict{String, Int}(
        "[UNK]" => 50280,
        "[CLS]" => 50281,
        "[SEP]" => 50282,
        "[PAD]" => 50283,
        "[MASK]" => 50284,
        "[CONT]" => 50285,
        "[END]" => 50288
    )
    
    # Initialize vocabulary
    vocab = Dict{String, Int}()
    
    println("\n1. Loading base vocabulary...")
    vocab_data = JSON3.read(read(vocab_path, String))
    if haskey(vocab_data, :model) && haskey(vocab_data.model, :vocab)
        for (token, id) in pairs(vocab_data.model.vocab)
            vocab[String(token)] = id
        end
    end
    println("Base vocabulary size: ", length(vocab))
    
    println("\n2. Adding special tokens...")
    for (token, id) in special_tokens
        vocab[token] = id
        println("Added $token => $id (now in vocab: $(haskey(vocab, token)))")
    end
    println("Vocabulary size after special tokens: ", length(vocab))
    
    println("\n3. Adding required tokens...")
    required_tokens = Dict{String, Int}(
        " " => 50275,
        "Ġ" => 50286,
        "Ċ" => 50287
    )
    
    for (token, id) in required_tokens
        if !haskey(vocab, token)
            vocab[token] = id
            println("Added $token => $id")
        else
            println("Token $token already exists with ID $(vocab[token])")
        end
    end
    
    println("\nFinal vocabulary size: ", length(vocab))
    println("\n=== Token Loading Complete ===")
end

vocab_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
debug_token_loading(vocab_path)
