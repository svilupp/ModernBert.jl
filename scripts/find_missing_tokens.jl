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

# Create sets for efficient lookup
base_tokens = Set(values(vocab))
special_ids = Set(values(special_tokens))
all_ids = union(base_tokens, special_ids)

println("Base vocabulary size: ", length(base_tokens))
println("Special tokens: ", length(special_tokens))
println("Total vocabulary size: ", length(all_ids))

# Check specific ranges for missing IDs
println("\nMissing IDs in base vocabulary range (0-50279):")
for i in 0:50279
    if i ∉ base_tokens
        println("Missing base token: ", i)
    end
end

println("\nMissing IDs in special token range (50280-50284):")
for i in 50280:50284
    if i ∉ special_ids
        println("Missing special token: ", i)
    end
end

println("\nMissing IDs in extended range (50285-50288):")
for i in 50285:50288
    if i ∉ all_ids
        println("Missing token: ", i)
    end
end

# Print special token mappings for verification
println("\nSpecial token mappings:")
for (token, id) in sort(collect(special_tokens))
    println("$token => $id")
end
