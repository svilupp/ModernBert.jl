using JSON3

# Read the tokenizer configuration
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path, String))

# Print structure overview
println("=== Tokenizer Configuration Analysis ===")

# Analyze vocabulary structure
println("\nConfiguration Structure:")
for key in keys(config)
    println("  $key")
end

if haskey(config, :model)
    println("\nModel Configuration Keys:")
    for key in keys(config.model)
        println("  $key")
    end
end

# Analyze vocabulary
if haskey(config.model, :vocab)
    vocab = config.model.vocab
    vocab_size = length(vocab)
    println("\nVocabulary Analysis:")
    println("Total vocabulary size: $vocab_size")

    # Token ID analysis
    token_ids = collect(values(vocab))
    min_id = minimum(token_ids)
    max_id = maximum(token_ids)
    zero_based = any(id == 0 for id in token_ids)
    negative = any(id < 0 for id in token_ids)
    
    println("\nToken ID Statistics:")
    println("  Minimum ID: $min_id")
    println("  Maximum ID: $max_id")
    println("  Contains zero-based IDs: $zero_based")
    println("  Contains negative IDs: $negative")
    println("  Number of unique IDs: $(length(unique(token_ids)))")

    # Special token verification
    special_tokens = Dict(
        "[CLS]" => 50281,
        "[SEP]" => 50282,
        "[MASK]" => 50284,
        "[PAD]" => 50283,
        "[UNK]" => 50280
    )

    println("\nSpecial Token Verification:")
    for (token, expected_id) in special_tokens
        actual_id = get(vocab, token, nothing)
        if actual_id === nothing
            println("❌ $token: Not found (Expected: $expected_id)")
        else
            match = actual_id == expected_id ? "✓" : "❌"
            println("$match $token: Got $actual_id (Expected: $expected_id)")
        end
    end

    # Sample regular tokens
    println("\nSample Regular Tokens (first 5):")
    count = 0
    for (token, id) in sort(collect(vocab), by=x->x[2])[1:min(5, vocab_size)]
        if !haskey(special_tokens, token)
            println("  $token => $id")
            count += 1
        end
    end
end

# Analyze tokenizer settings
println("\nTokenizer Settings:")
if haskey(config, :normalizer)
    println("Normalizer configuration:")
    println(JSON3.write(config.normalizer, 2))
end
if haskey(config, :pre_tokenizer)
    println("\nPre-tokenizer configuration:")
    println(JSON3.write(config.pre_tokenizer, 2))
end
