using JSON3

# Load tokenizer configuration
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path, String))

# Print pre-tokenizer configuration
println("Pre-tokenizer Configuration:")
println(JSON3.write(config.pre_tokenizer, 2))

# Print normalizer configuration
println("\nNormalizer Configuration:")
println(JSON3.write(config.normalizer, 2))

# Extract and print special tokens
println("\nSpecial Tokens from vocab:")
for (token, id) in sort(collect(pairs(config.model.vocab)), by=x->x[2])
    token_str = String(token)
    if (startswith(token_str, "[") && endswith(token_str, "]")) ||
       (startswith(token_str, "<|") && endswith(token_str, "|>"))
        println("$(repr(token_str)) => $id")
    end
end

# Print whitespace and control character tokens
println("\nWhitespace and Control Character Tokens:")
for (token, id) in sort(collect(pairs(config.model.vocab)), by=x->x[2])
    token_str = String(token)
    if any(isspace, token_str) || token_str == "Ġ" || token_str == "\u0120"
        println("$(repr(token_str)) => $id")
    end
end

# Print first few merge rules
println("\nFirst 10 Merge Rules:")
for (i, merge_rule) in enumerate(config.model.merges[1:10])
    println("$i: $merge_rule")
end

# Print model type and settings
println("\nModel Configuration:")
for field in fieldnames(typeof(config.model))
    if field != :vocab && field != :merges  # Skip large fields
        value = getfield(config.model, field)
        println("$field: $value")
    end
end
