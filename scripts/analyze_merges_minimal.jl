using JSON3

# Read tokenizer config
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path, String))

# Extract first few merges for testing
println("First 10 merges:")
for (i, merge_pair) in enumerate(config.model.merges[1:10])
    println("$i: $merge_pair")
end
