using JSON3

# Debug function to print token details
function print_token_details(vocab_path)
    println("Loading vocabulary from: ", vocab_path)
    
    vocab_data = JSON3.read(read(vocab_path, String))
    base_vocab = Dict{String, Int}()
    
    if haskey(vocab_data, :model) && haskey(vocab_data.model, :vocab)
        for (token, id) in pairs(vocab_data.model.vocab)
            base_vocab[String(token)] = id
        end
    end
    
    special_tokens = Dict{String, Int}(
        "[UNK]" => 50280,
        "[CLS]" => 50281,
        "[SEP]" => 50282,
        "[PAD]" => 50283,
        "[MASK]" => 50284
    )
    
    required_tokens = Dict{String, Int}(
        " " => 50275,
        "Ġ" => 50286,
        "Ċ" => 50287
    )
    
    println("\nBase vocabulary size: ", length(base_vocab))
    
    println("\nSpecial tokens:")
    for (token, id) in special_tokens
        println("  $token => $id (in base vocab: $(haskey(base_vocab, token)))")
    end
    
    println("\nRequired tokens:")
    for (token, id) in required_tokens
        println("  $token => $id (in base vocab: $(haskey(base_vocab, token)))")
    end
    
    # Add all tokens to get final size
    for (token, id) in special_tokens
        base_vocab[token] = id
    end
    for (token, id) in required_tokens
        base_vocab[token] = id
    end
    
    println("\nFinal vocabulary size: ", length(base_vocab))
end

vocab_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
print_token_details(vocab_path)
