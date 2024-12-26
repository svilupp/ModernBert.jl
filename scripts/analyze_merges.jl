using JSON3
using BytePairEncoding
using BytePairEncoding: Merge, BPE

# Load and parse the tokenizer configuration
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path, String))

# Analyze merges structure
println("=== Analyzing Merges ===")
println("Number of merges: ", length(config.model.merges))

# Print first few merges to understand structure
println("\nFirst 10 merges:")
for (i, merge_entry) in enumerate(Iterators.take(config.model.merges, 10))
    println("$i: $merge_entry")
end

# Create proper BPE merges dictionary
bpe_merges = Dict{Tuple{Merge, Merge}, Int}()
for (token, id) in config.model.merges
    token_str = string(token)
    parts = split(token_str)
    if length(parts) == 2
        merge_pair = (Merge(string(parts[1])), Merge(string(parts[2])))
        bpe_merges[merge_pair] = id
    end
end

println("\n=== BPE Merges Analysis ===")
println("Number of valid merge pairs: ", length(bpe_merges))

# Print first few processed merges
println("\nFirst 10 processed merges:")
for (i, (merge_pair, id)) in enumerate(Iterators.take(bpe_merges, 10))
    println("$i: $(merge_pair) => $id")
end
