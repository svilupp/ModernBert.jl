using JSON3

# Load config
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config_str = read(config_path, String)

# Parse config as raw JSON to access added_tokens field
config_raw = JSON3.read(config_str)

println("=== Added Tokens Analysis ===")

# Analyze added_tokens field
if haskey(config_raw, :added_tokens)
    added_tokens = config_raw.added_tokens
    println("\nNumber of added tokens: ", length(added_tokens))
    println("\nAdded tokens details:")
    for (i, token) in enumerate(added_tokens)
        println("\nToken $i:")
        for field in propertynames(token)
            println("  $field: $(token[field])")
        end
    end
else
    println("No added_tokens field in raw config")
end

# Check model vocabulary for these tokens
config = JSON3.read(config_str)
model_vocab = config.model.vocab
println("\n=== Model Vocabulary Analysis ===")
println("Model vocabulary size: ", length(model_vocab))

# Special token analysis
special_tokens = Dict(
    "[UNK]" => 50280,
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[PAD]" => 50283,
    "[MASK]" => 50284
)

println("\n=== Special Token Analysis ===")
for (token, expected_id) in special_tokens
    if haskey(model_vocab, token)
        println("$token found in model.vocab with ID: $(model_vocab[token])")
    else
        println("$token not found in model.vocab")
    end
end

# Check for any tokens with IDs >= 50280
println("\n=== High ID Token Analysis ===")
high_id_tokens = filter(pair -> pair.second >= 50280, collect(pairs(model_vocab)))
if !isempty(high_id_tokens)
    println("Found tokens with IDs ≥ 50280:")
    for (token, id) in high_id_tokens
        println("$token => $id")
    end
else
    println("No tokens found with IDs ≥ 50280")
end

# Check for any additional special-looking tokens
println("\n=== Additional Special Token Analysis ===")
special_chars = Set(['[', ']', '<', '>', '|'])
special_looking = filter(pair -> any(c -> c in special_chars, String(pair.first)), collect(pairs(model_vocab)))
if !isempty(special_looking)
    println("Found special-looking tokens:")
    for (token, id) in special_looking
        println("$token => $id")
    end
else
    println("No additional special-looking tokens found")
end
