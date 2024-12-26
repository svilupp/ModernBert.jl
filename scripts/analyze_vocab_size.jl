using JSON3
using ModernBert

# Load tokenizer configuration
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path, String))

# Analyze vocabulary
vocab = config.model.vocab
special_tokens = Dict{String,Int}(
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[MASK]" => 50284,
    "[PAD]" => 50283,
    "[UNK]" => 50280
)

# Count base vocabulary tokens
base_tokens = Set(values(vocab))
println("Base vocabulary size: ", length(base_tokens))
println("Special tokens: ", length(special_tokens))
println("Total vocabulary size: ", length(base_tokens) + length(special_tokens))

# Find gaps in token IDs
all_ids = sort(collect(union(base_tokens, values(special_tokens))))
for i in 0:50288
    if i ∉ all_ids
        println("Missing token ID: ", i)
    end
end

# Print special token mappings for verification
println("\nSpecial token mappings:")
for (token, id) in special_tokens
    println("$token => $id")
end
