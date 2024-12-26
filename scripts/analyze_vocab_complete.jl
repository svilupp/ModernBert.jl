using JSON3

# Load config
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path))

# Analyze vocabulary structure
println("Vocabulary Analysis:")
println("===================")

# Count model vocabulary tokens
model_vocab_size = length(config.model.vocab)
println("\nModel Vocabulary Size: ", model_vocab_size)

# Analyze token types
token_types = Dict{Type, Int}()
for (token, _) in config.model.vocab
    t = typeof(token)
    token_types[t] = get(token_types, t, 0) + 1
end

println("\nToken Types in Model Vocabulary:")
for (t, count) in token_types
    println("$t: $count tokens")
end

# Check for special tokens in config
println("\nSpecial Tokens in Config:")
if hasfield(typeof(config), :special_tokens)
    println(config.special_tokens)
else
    println("No special_tokens field in config")
end

# Check for additional tokens
println("\nChecking Additional Token Sources:")
for field in propertynames(config)
    if occursin("token", String(field))
        println("Found token-related field: $field")
    end
end

# Print first few tokens of each type
println("\nSample Tokens:")
for (t, _) in token_types
    tokens = filter(x -> typeof(x.first) == t, collect(pairs(config.model.vocab)))
    if !isempty(tokens)
        println("\nType $t samples:")
        for (token, id) in first(tokens, min(5, length(tokens)))
            println("  $token => $id")
        end
    end
end

# Check for any tokens with special prefixes
println("\nChecking for Special Prefixes:")
special_prefixes = ["Ġ", "Ċ", "ĉ", "Ń"]
for prefix in special_prefixes
    prefix_count = 0
    for token in keys(config.model.vocab)
        if startswith(String(token), prefix)
            prefix_count += 1
        end
    end
    if prefix_count > 0
        println("Prefix '$prefix': $prefix_count tokens")
    end
end

# Analyze added_tokens field
println("\nAnalyzing added_tokens field:")
if hasfield(typeof(config), :added_tokens)
    println("Number of added tokens: ", length(config.added_tokens))
    println("\nAdded tokens details:")
    for token in config.added_tokens
        println("Token: ", token)
    end
else
    println("No added_tokens field found in config")
end

# Look for potential missing tokens
println("\nPotential Missing Tokens:")
expected_special_tokens = Set(["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
found_special_tokens = Set{String}()

# Check model vocabulary for special tokens
for token in keys(config.model.vocab)
    token_str = String(token)
    if startswith(token_str, "[") && endswith(token_str, "]")
        push!(found_special_tokens, token_str)
    end
end

# Print missing special tokens
missing_special_tokens = setdiff(expected_special_tokens, found_special_tokens)
if !isempty(missing_special_tokens)
    println("Missing special tokens: ", join(missing_special_tokens, ", "))
end

# Additional special token analysis
println("\nSpecial Token Analysis:")
special_token_ids = Dict{String, Int}(
    "[UNK]" => 50280,
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[PAD]" => 50283,
    "[MASK]" => 50284
)

for (token, expected_id) in special_token_ids
    actual_id = get(config.model.vocab, token, nothing)
    if isnothing(actual_id)
        println("$token: Not found in model vocabulary")
    else
        println("$token: Found with ID $actual_id (Expected: $expected_id)")
    end
end
