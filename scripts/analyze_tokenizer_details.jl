using JSON3

# Load tokenizer configuration
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path, String))

# Analyze vocabulary
vocab = config.model.vocab
vocab_size = length(vocab)
println("Vocabulary size: ", vocab_size)

# Check special tokens
special_tokens = Dict(
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[MASK]" => 50284,
    "[PAD]" => 50283,
    "[UNK]" => 50280
)

println("\nSpecial Tokens Analysis:")
for (token, expected_id) in special_tokens
    actual_id = get(vocab, token, nothing)
    if actual_id === nothing
        println("$token: NOT FOUND (expected $expected_id)")
    else
        println("$token: $actual_id (expected $expected_id)")
    end
end

# Analyze merge rules
merges = config.model.merges
println("\nMerge Rules:")
println("Total merge rules: ", length(merges))
println("First 5 merge rules:")
for i in 1:min(5, length(merges))
    println(merges[i])
end

# Check pre-tokenizer configuration
println("\nPre-tokenizer Configuration:")
println(JSON3.write(config.pre_tokenizer, 2))

# Check normalizer configuration
println("\nNormalizer Configuration:")
println(JSON3.write(config.normalizer, 2))
