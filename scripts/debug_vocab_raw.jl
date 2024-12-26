using JSON3

# Load the vocabulary file
vocab_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
@assert isfile(vocab_path) "tokenizer.json not found"

println("Loading vocabulary file...")
data = JSON3.read(read(vocab_path, String))

println("\nRaw vocabulary structure:")
if haskey(data, :model)
    println("Has :model key")
    if haskey(data.model, :vocab)
        println("Has :vocab key in model")
        vocab_data = data.model.vocab
        println("Vocabulary size: ", length(vocab_data))
        
        # Check for special tokens in raw data
        special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        println("\nChecking for special tokens in raw vocabulary:")
        for token in special_tokens
            if haskey(vocab_data, Symbol(token))
                println("$token: ", vocab_data[Symbol(token)])
            else
                println("$token: not found in raw vocabulary")
            end
        end
        
        # Print first few entries to understand structure
        println("\nFirst few vocabulary entries:")
        count = 0
        for (token, id) in pairs(vocab_data)
            println("$token => $id")
            count += 1
            if count >= 5
                break
            end
        end
    else
        println("No :vocab key found in model")
    end
else
    println("No :model key found")
end

# Check added_tokens section if it exists
println("\nChecking for added_tokens section:")
if haskey(data, :added_tokens)
    added_tokens = data.added_tokens
    println("Found $(length(added_tokens)) added tokens")
    for token in added_tokens
        println("Token: ", token)
    end
else
    println("No added_tokens section found")
end
