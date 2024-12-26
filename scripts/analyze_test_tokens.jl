using BytePairEncoding
using TextEncodeBase
using JSON3

# Load GPT2 tokenizer
tkr = BytePairEncoding.load_gpt2()

# Load our vocabulary
vocab_file = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(vocab_file, String))
vocab = Dict(token => id for (token, id) in pairs(config.model.vocab))

# Test strings from test cases
text1 = "The capital of France is [MASK]."
text2 = "Hello world, how are you?"

println("=== Analyzing text1: \"$text1\" ===")
tokens1 = tkr(TextEncodeBase.Sentence(text1))
println("\nGPT2 Tokenization:")
for (i, token) in enumerate(tokens1)
    bytes = Vector{UInt8}(token)
    our_id = get(vocab, token, -1)
    println("Token $i: '$(token)' (bytes: $(bytes)) -> Our ID: $(our_id)")
end

println("\n=== Analyzing text2: \"$text2\" ===")
tokens2 = tkr(TextEncodeBase.Sentence(text2))
println("\nGPT2 Tokenization:")
for (i, token) in enumerate(tokens2)
    bytes = Vector{UInt8}(token)
    our_id = get(vocab, token, -1)
    println("Token $i: '$(token)' (bytes: $(bytes)) -> Our ID: $(our_id)")
end
