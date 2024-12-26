using ModernBert
using Test

# Initialize tokenizer
vocab_path = joinpath(dirname(@__DIR__), "test", "model", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"
tokenizer = load_modernbert_tokenizer(vocab_path)

# Test mixed text tokenization with simpler cases
mixed_text = [
    "Hello world!",
    "Testing spaces   and   punctuation.",
    "Simple @mention and #hashtag"
]

println("Running mixed text tokenization test...")

# Test each text individually first
for (idx, text) in enumerate(mixed_text)
    println("\nProcessing text $idx: \"$text\"")
    @time begin
        empty!(tokenizer.cache)
        tokens_single = encode(tokenizer, text)
        println("Single text tokens: ", tokens_single[1])
        println("Token count: ", length(tokens_single[1]))
    end
end

println("\nTesting batch processing...")
@time begin
    empty!(tokenizer.cache)
    tokens = encode(tokenizer, mixed_text)
    @test size(tokens[1], 2) == length(mixed_text)
    println("Test passed: Matrix dimensions match input size")
    println("Token matrix size: ", size(tokens[1]))
    
    # Print first few tokens of each text
    for i in 1:size(tokens[1], 2)
        println("Text $i first 5 tokens: ", tokens[1][1:min(5,end), i])
    end
end
