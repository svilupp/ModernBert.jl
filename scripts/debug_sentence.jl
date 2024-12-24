using JSON3
include("../src/bpe.jl")
include("../src/tokenizer.jl")

# Load tokenizer configuration
vocab_path = joinpath(@__DIR__, "..", "data", "tokenizer.json")

# Create tokenizer
bpe = create_bpe_tokenizer(vocab_path)

# Debug function to show sentence tokenization steps
function debug_sentence_tokenization(text::String)
    println("\nDebugging sentence tokenization for: '$text'")
    
    # Get list of special tokens
    special_token_list = collect(keys(bpe.special_tokens))
    
    # Show basic tokenization
    println("\n1. Basic tokenization:")
    basic_tokens = _bert_tokenise(text, Val(false), special_token_list)
    println("Basic tokens: ", basic_tokens)
    
    # Show BPE tokenization for each basic token
    println("\n2. BPE tokenization for each basic token:")
    all_bpe_tokens = String[]
    for (i, token) in enumerate(basic_tokens)
        println("\nProcessing token: '$token'")
        
        if haskey(bpe.special_tokens, token)
            println("Special token - keeping as is")
            push!(all_bpe_tokens, token)
            continue
        end
        
        if all(isspace, token)
            println("Whitespace token")
            space_tokens = bpe_encode(bpe, token)
            println("Space tokens: ", space_tokens)
            append!(all_bpe_tokens, space_tokens)
            continue
        end
        
        add_prefix = i > 1 && !all(isspace, basic_tokens[i-1])
        bpe_tokens = bpe_encode(bpe, token, add_prefix)
        println("Add prefix: $add_prefix")
        println("BPE tokens: ", bpe_tokens)
        append!(all_bpe_tokens, bpe_tokens)
    end
    
    println("\n3. Final tokens:")
    println(all_bpe_tokens)
    
    println("\n4. Token IDs:")
    token_ids = [get(bpe.special_tokens, t, get(bpe.vocab, t, bpe.special_tokens["[UNK]"])) for t in all_bpe_tokens]
    println(token_ids)
    return token_ids
end

# Test both test cases
println("\n=== Test Case 1 ===")
text1 = "The capital of France is [MASK]."
ids1 = debug_sentence_tokenization(text1)
println("\nExpected: [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282]")
println("Got:      $ids1")

println("\n=== Test Case 2 ===")
text2 = "Hello world! This is a test."
ids2 = debug_sentence_tokenization(text2)
println("\nExpected: [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282]")
println("Got:      $ids2")
