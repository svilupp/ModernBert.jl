using Test
using ModernBert: ModernBertEncoder, encode, tokenize

# Setup
vocab_path = joinpath(@__DIR__, "model", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"

# Create encoder
println("Creating encoder...")
encoder = ModernBertEncoder(vocab_path)
println("Encoder created successfully")

# Test basic tokenization
println("Testing basic tokenization...")
tokens = tokenize(encoder, "hello world")
@test tokens[1] == "[CLS]"
@test tokens[end] == "[SEP]"
println("Basic tokenization successful")

# Test basic encoding
println("Testing basic encoding...")
token_ids = encode(encoder, "hello world")
@test token_ids[1] == encoder.special_tokens["[CLS]"]
@test token_ids[end] == encoder.special_tokens["[SEP]"]
println("Basic encoding successful")

println("All tests completed successfully")
