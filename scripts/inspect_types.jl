using TextEncodeBase
using BytePairEncoding

# Import specific types
import TextEncodeBase: TokenStages, Token, Sentence

# Create test data
text = "Hello world"
sentence = Sentence(text)

# Create a simple token and token stages
token = Token{String, Nothing}(text, nothing)
tokens = [token]
token_stages = TokenStages{Token{String, Nothing}, Nothing}(tokens, nothing)

println("\nType Information:")
println("=================")
println("TokenStages:")
@show typeof(token_stages)
@show fieldnames(typeof(token_stages))

println("\nToken:")
@show typeof(token)
@show fieldnames(typeof(token))

println("\nSentence:")
@show typeof(sentence)
@show fieldnames(typeof(sentence))

# Test token mapping
println("\nToken Mapping Test:")
println("==================")
# Create a simple mapping
mapping = Dict(1 => 100, 2 => 200)
vocab = TextEncodeBase.Vocab(["[UNK]", "token1", "token2"], "[UNK]", 1)

# Create a TokenMapper instance with explicit type parameters
struct TestTokenMapper{T,M}
    mapping::Dict{Int, Int}
    vocab::TextEncodeBase.Vocab
end

mapper = TestTokenMapper{String,Int}(mapping, vocab)
@show typeof(mapper)
@show fieldnames(typeof(mapper))
