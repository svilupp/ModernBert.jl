using JSON3

# Load tokenizer config
config = JSON3.read(read("test/model/tokenizer.json", String))

# Count total vocabulary size
vocab_size = length(config.model.vocab)
println("Total vocabulary size: ", vocab_size)

# Check special tokens
special_tokens = Dict{String, Int}(
    "[UNK]" => 50280,
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[PAD]" => 50283,
    "[MASK]" => 50284
)

# Verify special tokens in vocabulary
for (token, expected_id) in special_tokens
    actual_id = get(config.model.vocab, Symbol(token), nothing)
    if actual_id == nothing
        println("Missing special token: ", token)
    elseif actual_id != expected_id
        println("Mismatched ID for token ", token, ": expected ", expected_id, " but got ", actual_id)
    else
        println("Correct special token mapping: ", token, " => ", actual_id)
    end
end

# Count tokens with Ġ prefix
gpt2_prefix_count = count(token -> startswith(String(token), "Ġ"), keys(config.model.vocab))
println("\nTokens with Ġ prefix: ", gpt2_prefix_count)

# Print some sample tokens around special token IDs
println("\nTokens around special token IDs:")
for id in 50275:50290
    token = findfirst(pair -> pair.second == id, config.model.vocab)
    if token !== nothing
        println("ID ", id, " => ", String(token))
    else
        println("ID ", id, " => not found")
    end
end
