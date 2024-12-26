using JSON3
using BytePairEncoding
using TextEncodeBase
using BytePairEncoding: GPT2Tokenization, Sentence, getvalue
using BytePairEncoding.GPT2: gpt2_codemap

# Load tokenizer configuration
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path, String))

# Print pre-tokenizer configuration
println("Pre-tokenizer Configuration:")
println(config.pre_tokenizer)

# Print normalizer configuration
println("\nNormalizer Configuration:")
println(config.normalizer)

# Create a small test case
test_text = "The capital of France is [MASK]."
println("\nTest text: ", test_text)

# Create base tokenizer components
base_tokenizer = GPT2Tokenization()
println("\nGPT2Tokenization settings:")
println("add_prefix_space: ", base_tokenizer.add_prefix_space)
println("trim_offsets: ", base_tokenizer.trim_offsets)

# Print token IDs for special characters
println("\nSpecial character token IDs:")
for char in [" ", "\n", "\t"]
    token = String(getvalue(first(base_tokenizer(Sentence(char)))))
    println("'$char' -> '$(token)'")
end

# Print first few merge rules
println("\nFirst 10 merge rules:")
for rule in first(config.model.merges, 10)
    println(rule)
end

# Print vocabulary entries around space tokens
println("\nVocabulary entries around space tokens:")
for (token, id) in config.model.vocab
    if occursin(r"^\s+$", String(token)) || startswith(String(token), "Ġ")
        println("'$(String(token))' => $id")
    end
end
