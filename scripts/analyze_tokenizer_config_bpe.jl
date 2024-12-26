using JSON3

# Read and parse tokenizer configuration
tokenizer_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(tokenizer_path, String))

# Extract vocabulary and merges
vocab = config.model.vocab
merges = config.model.merges

# Print configuration details
println("Tokenizer Configuration:")
println("=======================")
println("Model Type: ", config.model.type)
println("Vocabulary Size: ", length(vocab))
println("Number of Merges: ", length(merges))
println("\nNormalizer:")
println("Type: ", config.normalizer.type)
println("\nPre-tokenizer:")
println("Type: ", config.pre_tokenizer.type)
println("Add Prefix Space: ", config.pre_tokenizer.add_prefix_space)
println("Trim Offsets: ", config.pre_tokenizer.trim_offsets)
println("Use Regex: ", config.pre_tokenizer.use_regex)

# Print first few vocabulary entries
println("\nFirst 5 Vocabulary Entries:")
for (i, (token, id)) in enumerate(vocab)
    i > 5 && break
    println("$token => $id")
end

# Print first few merge rules
println("\nFirst 5 Merge Rules:")
for (i, merge) in enumerate(merges)
    i > 5 && break
    println(merge)
end

# Verify special tokens
special_tokens = Dict(
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[MASK]" => 50284,
    "[PAD]" => 50283,
    "[UNK]" => 50280
)

println("\nSpecial Tokens Verification:")
for (token, expected_id) in special_tokens
    actual_id = get(vocab, token, nothing)
    status = actual_id == expected_id ? "✓" : "✗"
    println("$status $token: Expected=$expected_id, Actual=$actual_id")
end
