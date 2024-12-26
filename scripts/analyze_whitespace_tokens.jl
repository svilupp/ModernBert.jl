using JSON3

# Load tokenizer configuration
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path, String))

# Extract vocabulary
vocab = Dict{String,Int}()
for (token, id) in pairs(config.model.vocab)
    vocab[String(token)] = id
end

# Function to convert token to readable format
function format_token(token)
    try
        # Try to make non-printable characters visible
        return repr(token)
    catch
        return "<unprintable>"
    end
end

# Find tokens that might be whitespace-related
whitespace_tokens = Dict{String,Int}()
for (token, id) in vocab
    if any(c -> isspace(c), token) || occursin(r"Ġ|▁", token)
        whitespace_tokens[token] = id
    end
end

# Sort by token ID
sorted_tokens = sort(collect(whitespace_tokens), by=x->x[2])

# Print findings
println("=== Whitespace-related tokens ===")
for (token, id) in sorted_tokens
    println("ID: $(id), Token: $(format_token(token))")
end

# Find token with ID 209
token_209 = nothing
for (token, id) in vocab
    if id == 209
        token_209 = token
        break
    end
end

if token_209 !== nothing
    println("\nToken with ID 209: $(format_token(token_209))")
    # Also print surrounding tokens
    println("\nSurrounding tokens:")
    for id in 207:211
        for (token, tid) in vocab
            if tid == id
                println("ID: $(id), Token: $(format_token(token))")
                break
            end
        end
    end
else
    println("\nNo token found with ID 209")
end
