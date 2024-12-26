using JSON3

# Load tokenizer configuration
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path, String))

# Print the top-level structure
println("Top-level keys:")
for key in keys(config)
    println("- $key")
end

# If there's a model key, print its structure
if haskey(config, :model)
    println("\nModel-level keys:")
    for key in keys(config.model)
        println("- $key")
    end
end

# Print first few entries of any vocabulary found
if haskey(config, :model) && haskey(config.model, :vocab)
    println("\nFirst few vocabulary entries:")
    vocab_entries = collect(pairs(config.model.vocab))
    for (token, id) in sort(vocab_entries[1:min(5, length(vocab_entries))], by=x->x[2])
        println("$(repr(token)) => $id")
    end
end
