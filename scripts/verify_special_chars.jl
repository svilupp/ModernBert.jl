include("../src/minimal_tokenizer.jl")
using .ModernBertTokenizerImpl

# Create minimal vocabulary for test cases
vocab = Dict{String,Int}(
    "H" => 29,
    "₂" => 209,
    "O" => 47,
    "~" => 187,
    "7" => 17,
    "." => 13,
    "0" => 16,
    "你" => 4821,
    "好" => 4928
)

# Load special tokens
special_tokens = Dict{String,Int}(
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[PAD]" => 50283,
    "[MASK]" => 50284,
    "[UNK]" => 50280
)

# Initialize tokenizer with minimal test cases
tokenizer = ModernBertTokenizer(vocab, special_tokens)

# Reduce test cases for performance
test_cases = [
    "H₂O",           # Focus on subscript handling
    "~7.0",          # Special character case
    "你好"            # UTF-8 case
]

# Test cases already defined above

println("Running test cases with timing:")
for (i, text) in enumerate(test_cases)
    println("\nCase $i: $text")
    # Run multiple times to get stable timing
    tokens = Int[]
    min_time = Inf
    for _ in 1:5
        elapsed = @elapsed tokens = tokenize(tokenizer, text)
        min_time = min(min_time, elapsed)
    end
    println("Tokens: ", tokens)
    println("Best Time: $(min_time*1000) ms")
    
    # Print token details for debugging
    println("Token details:")
    for token_id in tokens
        token = get(tokenizer.id_to_token, token_id, "[Unknown]")
        println("  $token_id => $token")
    end
end
