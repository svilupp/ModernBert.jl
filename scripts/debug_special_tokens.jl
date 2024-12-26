using JSON3
using ModernBert

# Load config
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path))

# Initialize vocabularies
model_tokens = Dict{String, Int}()
special_tokens = Dict{String, Int}(
    "[UNK]" => 50280,
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[PAD]" => 50283,
    "[MASK]" => 50284
)

# Load model vocabulary
for (token, id) in pairs(config.model.vocab)
    token_str = String(token)
    model_tokens[token_str] = id
end

println("Initial model vocabulary size: ", length(model_tokens))

# Create final vocabulary
final_vocab = copy(model_tokens)

# Add special tokens
println("\nAdding special tokens:")
for (token, id) in special_tokens
    if haskey(final_vocab, token)
        println("WARNING: Special token '$token' already exists with ID $(final_vocab[token])")
        println("Overwriting with ID $id")
    end
    final_vocab[token] = id
    println("Added '$token' with ID $id")
end

println("\nFinal vocabulary size: ", length(final_vocab))

# Verify special tokens
println("\nVerifying special tokens in final vocabulary:")
for (token, expected_id) in special_tokens
    actual_id = get(final_vocab, token, nothing)
    if actual_id === nothing
        println("ERROR: Special token '$token' is missing")
    elseif actual_id != expected_id
        println("ERROR: Special token '$token' has wrong ID: $actual_id (expected $expected_id)")
    else
        println("OK: Special token '$token' has correct ID: $actual_id")
    end
end

# Check for any tokens with IDs in special token range
println("\nChecking for conflicts in special token ID range:")
for (token, id) in final_vocab
    if id ≥ 50280 && id ≤ 50284 && !haskey(special_tokens, token)
        println("WARNING: Found non-special token '$token' with ID in special range: $id")
    end
end

# Check for duplicate tokens (same token string, different IDs)
println("\nChecking for duplicate tokens:")
token_to_ids = Dict{String, Vector{Int}}()
for (token, id) in final_vocab
    if !haskey(token_to_ids, token)
        token_to_ids[token] = Int[]
    end
    push!(token_to_ids[token], id)
end

for (token, ids) in token_to_ids
    if length(ids) > 1
        println("DUPLICATE: Token '$token' has multiple IDs: $ids")
    end
end
