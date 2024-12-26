using JSON3

# Load tokenizer configuration
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path, String))

# Extract and print pre-tokenizer settings
println("Pre-tokenizer Configuration:")
println(JSON3.write(config.model.pre_tokenizer, 2))

# Extract and print normalizer settings
println("\nNormalizer Configuration:")
println(JSON3.write(config.model.normalizer, 2))

# Extract special tokens and their IDs
special_tokens = Dict{String,Int}()
for (token, id) in config.model.vocab
    if startswith(token, "[") && endswith(token, "]")
        special_tokens[String(token)] = id
    end
end

println("\nSpecial Tokens:")
for (token, id) in sort(collect(special_tokens), by=x->x[2])
    println("$token => $id")
end

# Print first few vocabulary entries to check mapping
println("\nFirst 10 Regular Vocabulary Entries:")
regular_tokens = filter(p -> !startswith(p.first, "["), config.model.vocab)
for (token, id) in first(sort(collect(regular_tokens), by=x->x[2]), 10)
    println("$(repr(token)) => $id")
end

# Check for whitespace-related tokens
println("\nWhitespace-related Tokens:")
for (token, id) in sort(collect(config.model.vocab), by=x->x[2])
    if any(isspace, token) || token == "Ġ" || token == "\u0120"
        println("$(repr(token)) => $id")
    end
end
