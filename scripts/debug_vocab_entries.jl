using ModernBert
using JSON3

# Load tokenizer
vocab_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"
tokenizer = load_modernbert_tokenizer(vocab_path)

# Print vocabulary size
println("Vocabulary size: ", length(tokenizer.vocab))

# Check specific tokens from test case
test_tokens = ["The", " The", "capital", " capital"]
println("\nTest case tokens:")
for token in test_tokens
    println("Token: \"", token, "\" -> ", get(tokenizer.vocab, token, "not found"))
end

# Print first few vocabulary entries
println("\nFirst 10 vocabulary entries:")
let count = 0
    for (token, id) in tokenizer.vocab
        println("\"", token, "\" -> ", id)
        count += 1
        count >= 10 && break
    end
end

# Print space-prefixed tokens
println("\nSpace-prefixed tokens:")
space_tokens = filter(pair -> startswith(pair.first, " "), tokenizer.vocab)
for (i, (token, id)) in enumerate(space_tokens)
    println("\"", token, "\" -> ", id)
    i >= 10 && break
end

# Print special tokens
println("\nSpecial tokens:")
for (token, id) in tokenizer.special_tokens
    println("\"", token, "\" -> ", id)
end
