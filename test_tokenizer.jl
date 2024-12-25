using ModernBert, JSON3

# Check model files and configuration
println("\nChecking model files and configuration...")
vocab_path = joinpath(@__DIR__, "data", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"

# Load and compare tokenizer configuration
println("\nLoading and comparing tokenizer configuration...")
config = JSON3.read(read(vocab_path))
println("Vocabulary size: ", length(config["model"]["vocab"]))
special_tokens = [token["content"] for token in config["added_tokens"] if get(token, "special", false)]
println("Special tokens: ", special_tokens)
@assert length(special_tokens) >= 5 "Missing required special tokens"

# Create BertTextEncoder
println("\nCreating encoder...")
bpe = create_bpe_tokenizer(vocab_path)

# Create vocabulary from config
vocab = Dict{String,Int}()
for (token, id) in config["model"]["vocab"]
    vocab[String(token)] = id
end

# Add special tokens to vocabulary
for token in config["added_tokens"]
    content = String(token["content"])
    id = Int(token["id"])
    vocab[content] = id
end

encoder = BertTextEncoder(bpe, vocab)

# Test specific token sequences
println("\nTesting specific token sequences...")

# Test case 1
text1 = "The capital of France is [MASK]."
expected_ids1 = [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282]
tokens1 = tokenize(encoder, text1; token_ids=true)
println("\nTest case 1:")
println("Input: ", text1)
println("Expected: ", expected_ids1)
println("Got: ", tokens1)

# Debug output for token sequence
println("\nDetailed token analysis:")
for (i, (got, expected)) in enumerate(zip(tokens1, expected_ids1))
    word_start = i > 1 ? text1[findnext(isspace, text1, 1):end] : text1
    println("Position $i:")
    println("  Text context: ", word_start[1:min(end, 20)], "...")
    println("  Got token ID: ", got)
    println("  Expected ID: ", expected)
    if got != expected
        # Try to find the actual token for these IDs
        got_token = nothing
        expected_token = nothing
        for (token, id) in encoder.tokenizer.vocab
            if id == got
                got_token = token
            end
            if id == expected
                expected_token = token
            end
            if !isnothing(got_token) && !isnothing(expected_token)
                break
            end
        end
        println("  Got token: ", got_token)
        println("  Expected token: ", expected_token)
        println("  Previous token: ", encoder.tokenizer.prev_token)
    end
end

@assert tokens1 == expected_ids1 "Token sequence 1 does not match expected output"

# Test case 2
text2 = "Hello world! This is a test."
expected_ids2 = [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282]
tokens2 = tokenize(encoder, text2; token_ids=true)
println("\nTest case 2:")
println("Input: ", text2)
println("Expected: ", expected_ids2)
println("Got: ", tokens2)
@assert tokens2 == expected_ids2 "Token sequence 2 does not match expected output"

# Test case 3: Complex case with British/American English variants
text3 = "The antidisestablishmentarianistically-minded pseudopneumonoultramicroscopicsilicovolcanoconiosis researcher hypothesized that supercalifragilisticexpialidocious manifestations of hippopotomonstrosesquippedaliophobia could be counterrevolutionarily interconnected with floccinaucinihilipilification tendencies among immunoelectrophoretically-sensitive microspectrophotofluorometrically-analyzed organophosphates..."
expected_ids3 = [50281, 510, 1331, 30861, 15425, 8922, 6656, 18260, 14, 23674, 10585, 412, 11312, 251, 3941, 1206, 312, 1813, 1026, 412, 982, 24479, 729, 311, 5092, 406, 12355, 2433, 22780, 24045, 326, 2221, 1179, 338, 22194, 300, 382, 547, 37755, 451, 301, 49514, 26683, 273, 15241, 43191, 16142, 296, 2921, 265, 371, 6390, 8952, 40988, 812, 320, 4828, 15079, 2241, 3441, 36282, 342, 892, 34295, 1758, 1028, 5391, 73, 300, 532, 300, 1877, 37725, 2190, 2984, 41121, 3838, 7262, 1037, 14, 19579, 2494, 18181, 36293, 21158, 263, 2755, 16671, 14, 29965, 4337, 1963, 25982, 684, 1051, 50282]
println("\nTest case 3:")
println("Input: ", text3)
println("Expected: ", expected_ids3)
println("\nStarting tokenization of complex word...")
tokens3 = tokenize(encoder, text3; token_ids=true)
println("Tokenization completed")
println("Got: ", tokens3)

# Compare results and show mismatches
if tokens3 == expected_ids3
    println("✓ Test case 3 passed")
else
    println("✗ Test case 3 failed")
    println("\nMismatched tokens:")
    println("Index\tGot\tExpected")
    println("-------------------------")
    for (i, (got, expected)) in enumerate(zip(tokens3, expected_ids3))
        if got != expected
            println("$i\t$got\t$expected")
        end
    end
end
@assert tokens3 == expected_ids3 "Token sequence 3 does not match expected output"

println("\nTokenizer tests completed successfully!")
