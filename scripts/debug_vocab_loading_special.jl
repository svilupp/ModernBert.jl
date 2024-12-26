using JSON3

# Load the vocabulary file
vocab_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"

println("Loading vocabulary file...")
data = JSON3.read(read(vocab_path, String))

# Initialize vocabulary
vocab = Dict{String, Int}()

# Define special tokens
special_tokens = Dict{String, Int}(
    "[UNK]" => 50280,
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[PAD]" => 50283,
    "[MASK]" => 50284
)

# Define required tokens
required_tokens = Dict{String, Int}(
    " " => 50275,  # space token
    "Ġ" => 50286,  # GPT-2 space token
    "Ċ" => 50287   # GPT-2 newline token
)

println("\nAdding special tokens...")
for (token, id) in special_tokens
    vocab[token] = id
    println("Added special token: $token => $id")
end

println("\nAdding required tokens...")
for (token, id) in required_tokens
    vocab[token] = id
    println("Added required token: $token => $id")
end

println("\nLoading main vocabulary...")
if haskey(data, :model) && haskey(data.model, :vocab)
    for (token, id) in pairs(data.model.vocab)
        str_token = String(token)
        # Only add if not already present (preserve special and required tokens)
        if !haskey(vocab, str_token)
            vocab[str_token] = id
        end
    end
end

println("\nVerifying special tokens after loading...")
for (token, expected_id) in special_tokens
    actual_id = get(vocab, token, nothing)
    println("Token: $token")
    println("  Expected ID: $expected_id")
    println("  Actual ID: $actual_id")
    println("  Match: $(actual_id == expected_id)")
    println()
end

println("\nVerifying required tokens after loading...")
for (token, expected_id) in required_tokens
    actual_id = get(vocab, token, nothing)
    println("Token: $token")
    println("  Expected ID: $expected_id")
    println("  Actual ID: $actual_id")
    println("  Match: $(actual_id == expected_id)")
    println()
end

println("\nVocabulary size: $(length(vocab))")
