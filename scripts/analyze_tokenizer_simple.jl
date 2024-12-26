using JSON3

# Load tokenizer configuration
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path, String))

# Print pre-tokenizer configuration
println("Pre-tokenizer Configuration:")
println(config.pre_tokenizer)

# Print normalizer configuration
println("\nNormalizer Configuration:")
println(config.normalizer)

# Extract and print special tokens
println("\nSpecial Tokens from vocab:")
special_tokens = Dict{String,Int}()
for (token, id) in pairs(config.model.vocab)
    token_str = String(token)
    if (startswith(token_str, "[") && endswith(token_str, "]")) ||
       (startswith(token_str, "<|") && endswith(token_str, "|>"))
        special_tokens[token_str] = id
    end
end

for (token, id) in sort(collect(special_tokens), by=x->x[2])
    println("$token => $id")
end

# Print model type and basic settings
println("\nModel Configuration:")
println("type: ", config.model.type)
println("unk_token: ", config.model.unk_token)
println("continuing_subword_prefix: ", config.model.continuing_subword_prefix)
println("end_of_word_suffix: ", config.model.end_of_word_suffix)

# Print first few merge rules
println("\nFirst 5 Merge Rules:")
for rule in first(config.model.merges, 5)
    println(rule)
end
