using JSON3

function analyze_vocab_structure()
    # Load config
    config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
    config = JSON3.read(read(config_path, String))
    
    # Special token IDs we expect
    special_tokens = Dict{String, Int}(
        "[UNK]" => 50280,
        "[CLS]" => 50281,
        "[SEP]" => 50282,
        "[PAD]" => 50283,
        "[MASK]" => 50284
    )
    
    # Analyze vocabulary structure
    println("=== Vocabulary Analysis ===")
    
    # Check model vocabulary
    model_vocab = config.model.vocab
    println("\nModel Vocabulary Stats:")
    println("Total entries: ", length(model_vocab))
    
    # Find highest and lowest IDs
    ids = [id for (_, id) in pairs(model_vocab)]
    println("ID range: ", minimum(ids), " to ", maximum(ids))
    
    # Check for special tokens in model vocab
    println("\nSpecial Tokens in Model Vocab:")
    for (token, id) in pairs(model_vocab)
        token_str = String(token)
        if startswith(token_str, "[") && endswith(token_str, "]")
            println("$token_str => $id")
        end
    end
    
    # Check for byte-level tokens
    byte_tokens = Dict{String, Int}()
    for (token, id) in pairs(model_vocab)
        token_str = String(token)
        if startswith(token_str, "Ġ") || startswith(token_str, "Ċ") || startswith(token_str, "ĉ")
            byte_tokens[token_str] = id
        end
    end
    println("\nByte-level Token Stats:")
    println("Total byte-level tokens: ", length(byte_tokens))
    println("Sample of byte-level tokens:")
    for (token, id) in Iterators.take(sort(collect(byte_tokens)), 5)
        println("$token => $id")
    end
    
    # Check for potential missing tokens
    println("\nPotential Missing Tokens:")
    all_ids = Set(ids)
    gaps = Int[]
    for i in 0:maximum(ids)
        if i ∉ all_ids
            push!(gaps, i)
        end
    end
    println("Number of gaps in ID sequence: ", length(gaps))
    if !isempty(gaps)
        println("First few gaps: ", gaps[1:min(5, length(gaps))])
    end
    
    # Additional token categories
    single_chars = Dict{String, Int}()
    for (token, id) in pairs(model_vocab)
        token_str = String(token)
        if length(token_str) == 1
            single_chars[token_str] = id
        end
    end
    println("\nSingle Character Tokens:")
    println("Total single character tokens: ", length(single_chars))
    println("Sample of single character tokens:")
    for (token, id) in Iterators.take(sort(collect(single_chars)), 5)
        println("$token => $id")
    end
end

analyze_vocab_structure()
