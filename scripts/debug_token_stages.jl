using TextEncodeBase
using BytePairEncoding
using ModernBert

# Create a simple sentence
sentence = TextEncodeBase.Sentence("Hello")

# Create a basic tokenizer
tokenizer = BytePairEncoding.ByteLevelPreTokenization()

# Try to wrap the sentence
result = TextEncodeBase.wrap(tokenizer, sentence)

# Print type information
println("Type information:")
println("sentence type: ", typeof(sentence))
println("result type: ", typeof(result))

# Try our TokenMapper
vocab = Dict{String, Int}()
mapping = Dict{Int, Int}()
tm = ModernBert.TokenMapper(mapping, vocab)

# Try to wrap with our TokenMapper
wrapped = TextEncodeBase.wrap(tm, sentence)
println("wrapped type: ", typeof(wrapped))
