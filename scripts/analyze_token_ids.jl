using JSON3
using ModernBert

# Load tokenizer config
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = open(config_path, "r") do f
    JSON3.read(f)
end

# Expected token IDs from test
expected_tokens = Dict(
    "The" => 510,
    "capital" => 5347,
    "of" => 273,
    "France" => 6181,
    "is" => 310
)

# Print vocabulary entries for analysis
println("\nAnalyzing vocabulary mappings...")
for (token, expected_id) in expected_tokens
    # Check direct token
    direct_id = get(config.model.vocab, token, nothing)
    println("\nToken: '$token'")
    println("Expected ID: $expected_id")
    println("Direct ID: $direct_id")
    
    # Check with Ġ prefix
    prefix_token = "Ġ" * token
    prefix_id = get(config.model.vocab, prefix_token, nothing)
    println("Prefix token: '$prefix_token'")
    println("Prefix ID: $prefix_id")
end

# Print relevant merge rules
println("\nAnalyzing merge rules...")
for merge in config.model.merges[1:10]
    println(merge)
end
