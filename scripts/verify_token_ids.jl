using JSON3
using ModernBert

# Load the vocabulary file
vocab_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"

data = JSON3.read(open(vocab_path))
vocab = data.model.vocab

# Expected token mappings from test
expected_tokens = Dict(
    "The" => 510,
    "capital" => 38479,
    "of" => 1171,
    "France" => 33639,
    "is" => 261,
    "." => 15
)

# Check if tokens exist with correct IDs
println("Checking token mappings...")
for (token, expected_id) in expected_tokens
    # Try both with and without Ġ prefix
    found = false
    actual_id = nothing
    
    # Check regular token
    if haskey(vocab, token)
        actual_id = vocab[token]
        found = true
    end
    
    # Check with Ġ prefix
    if !found && haskey(vocab, "Ġ" * token)
        actual_id = vocab["Ġ" * token]
        found = true
    end
    
    if found
        println("Token: $(token)")
        println("  Expected ID: $(expected_id)")
        println("  Actual ID: $(actual_id)")
        println("  Match: $(actual_id == expected_id)")
    else
        println("Token '$(token)' not found in vocabulary!")
    end
    println()
end

# Also check special tokens
special_tokens = Dict(
    "[UNK]" => 50280,
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[PAD]" => 50283,
    "[MASK]" => 50284
)

println("\nChecking special tokens...")
for (token, expected_id) in special_tokens
    actual_id = get(vocab, token, nothing)
    println("Token: $(token)")
    println("  Expected ID: $(expected_id)")
    println("  Actual ID: $(actual_id)")
    println("  Match: $(actual_id == expected_id)")
    println()
end
