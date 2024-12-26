using JSON3
using ModernBert

# Load vocabulary
vocab_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"

# Load and examine vocabulary structure
vocab_json = JSON3.read(read(vocab_path, String))
println("Vocabulary type: ", typeof(vocab_json.model.vocab))
println("\nSample entries:")
for (token, id) in Iterators.take(vocab_json.model.vocab, 5)
    println("Token: '", token, "' (type: ", typeof(token), "), ID: ", id, " (type: ", typeof(id), ")")
end

# Check specific tokens
tokens_to_check = ["The", "Hello", "Mr.", "Ġ", "ĠThe", "ĠHello", "ĠMr."]
println("\nChecking specific tokens:")
for token in tokens_to_check
    id = get(vocab_json.model.vocab, token, nothing)
    sym_id = get(vocab_json.model.vocab, Symbol(token), nothing)
    println("Token: '", token, "'")
    println("  As string: ", id)
    println("  As symbol: ", sym_id)
end
