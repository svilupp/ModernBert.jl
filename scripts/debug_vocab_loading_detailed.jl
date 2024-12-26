using JSON3
using ModernBert

# Load config
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path))

# Analyze model vocabulary
model_tokens = Dict{String, Int}()
for (token, id) in pairs(config.model.vocab)
    token_str = String(token)
    model_tokens[token_str] = id
end

# Print statistics
println("Model vocabulary statistics:")
println("Total tokens: ", length(model_tokens))
println("\nFirst 10 tokens:")
for (i, (token, id)) in enumerate(first(model_tokens, 10))
    println("$i. '$token' => $id")
end

println("\nLast 10 tokens:")
for (i, (token, id)) in enumerate(Iterators.take(Iterators.reverse(collect(model_tokens)), 10))
    println("$i. '$token' => $id")
end

# Check for special token conflicts
special_tokens = Dict{String, Int}(
    "[UNK]" => 50280,
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[PAD]" => 50283,
    "[MASK]" => 50284
)

println("\nChecking for special token conflicts:")
for (token, id) in special_tokens
    if haskey(model_tokens, token)
        println("WARNING: Special token '$token' exists in model vocabulary with ID $(model_tokens[token])")
    end
end

# Check ID ranges
println("\nID statistics:")
println("Min ID: ", minimum(values(model_tokens)))
println("Max ID: ", maximum(values(model_tokens)))

# Check for gaps in IDs
all_ids = Set(values(model_tokens))
expected_ids = Set(0:50279)
missing_ids = setdiff(expected_ids, all_ids)
extra_ids = setdiff(all_ids, expected_ids)

println("\nMissing IDs: ", length(missing_ids) > 0 ? collect(missing_ids) : "None")
println("Extra IDs: ", length(extra_ids) > 0 ? collect(extra_ids) : "None")
