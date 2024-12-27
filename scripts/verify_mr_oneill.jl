include("../src/ModernBert.jl")
using .ModernBertTokenizerImpl

# Initialize tokenizer with empty vocabulary
tokenizer = load_modernbert_tokenizer()

# Add test vocabulary entries
tokenizer.known_tokens["Mr"] = 7710
tokenizer.known_tokens["ĠMr"] = 7710
tokenizer.known_tokens["ĠO"] = 473
tokenizer.known_tokens["Neill"] = 41437
tokenizer.known_tokens["Mc"] = 11773
tokenizer.known_tokens["."] = 15
tokenizer.known_tokens["'"] = 8
tokenizer.known_tokens["-"] = 14
tokenizer.known_tokens["Pherson"] = 29845
tokenizer.known_tokens["'s"] = 502

# Test cases
test_cases = [
    ("Mr.", ["Mr", "."]),  # Should be either ["Mr", "."] or ["Mr."]
    ("O'Neill", ["ĠO", "'", "Neill"]),  # Should start with "ĠO" (ID 473)
    ("Mr. O'Neill-McPherson's", ["Mr", ".", "ĠO", "'", "Neill", "-", "Mc", "Pherson", "'s"])
]

println("Running tokenization tests...")
for (text, expected_tokens) in test_cases
    println("\nTesting: ", text)
    tokens = tokenize(tokenizer, text)
    
    # Get token strings for debugging
    token_strs = [get(tokenizer.id_to_token, id, "<unknown>") for id in tokens]
    
    println("Got tokens: ", token_strs)
    println("Token IDs: ", tokens)
    
    # Special check for "ĠO"
    if "O'Neill" in text
        o_index = findfirst(==(473), tokens)
        if o_index !== nothing
            println("✓ Found 'ĠO' with correct ID 473")
        else
            println("✗ Missing 'ĠO' with ID 473")
        end
    end
    
    # Check for infinite loop with "ĠMr"
    if "Mr" in text
        mr_count = count(==(7710), tokens)
        if mr_count > 1
            println("✗ Found multiple 'Mr' tokens - possible loop detected")
        else
            println("✓ No token loop detected")
        end
    end
end
