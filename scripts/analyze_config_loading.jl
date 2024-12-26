using JSON3
using ModernBert

# Load config
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path))

println("Config structure:")
println("Keys in config: ", keys(config))
println("\nKeys in config.model: ", keys(config.model))

println("\nVocabulary analysis:")
println("Number of tokens in config.model.vocab: ", length(config.model.vocab))

# Check for additional token sources
if hasfield(typeof(config), :added_tokens)
    println("\nAdded tokens:")
    for token in config.added_tokens
        println("Token: ", token)
    end
end

# Check token types
println("\nToken type analysis:")
token_types = Dict{Type, Int}()
for (token, _) in config.model.vocab
    t = typeof(token)
    token_types[t] = get(token_types, t, 0) + 1
end
for (t, count) in token_types
    println("$t: $count tokens")
end

# Check for any special handling in config
println("\nSpecial token config:")
if hasfield(typeof(config), :special_tokens)
    println("Special tokens in config: ", config.special_tokens)
end

# Check for any additional vocabulary sources
println("\nChecking additional vocabulary sources:")
for field in propertynames(config)
    if occursin("vocab", String(field))
        println("Found vocab-related field: $field")
    end
end
