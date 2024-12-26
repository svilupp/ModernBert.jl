using JSON3

# Load config
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path))

println("=== Config Structure Analysis ===")

# Analyze model field
println("\n=== Model Field ===")
println("Model vocabulary size: ", length(config.model.vocab))
println("Model field keys: ", join(propertynames(config.model), ", "))

# Analyze added_tokens field
println("\n=== Added Tokens Field ===")
if hasfield(typeof(config), :added_tokens)
    added_tokens = config.added_tokens
    println("Added tokens count: ", length(added_tokens))
    println("\nAdded tokens details:")
    for (i, token) in enumerate(added_tokens)
        println("Token $i:")
        for field in propertynames(token)
            println("  $field: $(token[field])")
        end
    end
else
    println("No added_tokens field found")
end

# Check for special tokens in model vocabulary
println("\n=== Special Tokens in Model Vocabulary ===")
special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
for token in special_tokens
    if haskey(config.model.vocab, token)
        println("$token found in model.vocab with ID: $(config.model.vocab[token])")
    else
        println("$token not found in model.vocab")
    end
end

# Check for any additional special tokens
println("\n=== Additional Special Tokens ===")
bracket_tokens = filter(t -> startswith(String(t), "[") && endswith(String(t), "]"), 
                      collect(keys(config.model.vocab)))
println("All tokens with brackets:")
for token in bracket_tokens
    println("$token => $(config.model.vocab[token])")
end

# Analyze token ID ranges
println("\n=== Token ID Analysis ===")
ids = collect(values(config.model.vocab))
println("Min ID: ", minimum(ids))
println("Max ID: ", maximum(ids))
println("Number of unique IDs: ", length(unique(ids)))

# Check for any gaps in token IDs
sorted_ids = sort(unique(ids))
if length(sorted_ids) > 1
    gaps = findall(diff(sorted_ids) .> 1)
    if !isempty(gaps)
        println("\nFound gaps in token IDs:")
        for gap_idx in gaps
            println("Gap between $(sorted_ids[gap_idx]) and $(sorted_ids[gap_idx + 1])")
        end
    else
        println("\nNo gaps found in token IDs")
    end
end
