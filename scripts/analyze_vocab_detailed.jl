using JSON3

# Load tokenizer config
config = JSON3.read(read("test/model/tokenizer.json", String))

# Count total vocabulary size
vocab_size = length(config.model.vocab)
println("Total vocabulary size in config: ", vocab_size)

# Define special tokens and their expected IDs
special_tokens = Dict{String, Int}(
    "[UNK]" => 50280,
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[PAD]" => 50283,
    "[MASK]" => 50284
)

# Create a set of all IDs in the config
config_ids = Set(values(config.model.vocab))

# Check for tokens in the range of special tokens
println("\nTokens in special token ID range (50275-50290):")
for id in 50275:50290
    token = findfirst(pair -> pair.second == id, config.model.vocab)
    if token !== nothing
        println("ID ", id, " => ", String(token))
    end
end

# Find any gaps in the vocabulary IDs
println("\nGaps in vocabulary IDs:")
all_ids = sort(collect(config_ids))
if !isempty(all_ids)
    for i in 1:length(all_ids)-1
        if all_ids[i+1] - all_ids[i] > 1
            println("Gap between ", all_ids[i], " and ", all_ids[i+1])
        end
    end
end

# Print the highest ID in the vocabulary
if !isempty(all_ids)
    println("\nHighest ID in vocabulary: ", maximum(all_ids))
end

# Check for any tokens that might conflict with special token IDs
println("\nTokens that might conflict with special tokens:")
for (token, id) in special_tokens
    if any(vid == id for vid in values(config.model.vocab))
        conflicting_token = findfirst(pair -> pair.second == id, config.model.vocab)
        println("Conflict: ID ", id, " is used by both '", token, "' and '", String(conflicting_token), "'")
    end
end
