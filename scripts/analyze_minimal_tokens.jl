using JSON3

# Read the tokenizer.json file
vocab_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
data = open(vocab_path, "r") do f
    JSON3.read(f)
end

# Extract and verify special tokens
special_tokens = Dict{String,Int}(
    "[UNK]" => 50280,
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[PAD]" => 50283,
    "[MASK]" => 50284
)

# Extract test case tokens
test_tokens = Dict{String,Int}(
    "The" => 510,
    "capital" => 5347,
    "of" => 273,
    "France" => 6181,
    "is" => 310,
    "." => 15
)

println("Checking special tokens...")
vocab = data.model.vocab
for (token, expected_id) in special_tokens
    actual_id = get(vocab, token, nothing)
    println("$token: expected=$expected_id, actual=$actual_id")
end

println("\nChecking test case tokens...")
for (token, expected_id) in test_tokens
    actual_id = get(vocab, token, nothing)
    println("$token: expected=$expected_id, actual=$actual_id")
end
