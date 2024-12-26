using JSON3
using BytePairEncoding: Merge, BPE, GPT2Tokenization
using TextEncodeBase: FlatTokenizer, CodeNormalizer

"""
Analyze tokenizer.json and convert it to BytePairEncoding format.
This script helps understand the structure and create the correct conversion logic.
"""

function analyze_tokenizer_config()
    config_path = joinpath(@__DIR__, "..", "data", "tokenizer.json")
    config = JSON3.read(read(config_path, String))
    
    # Print tokenizer type and settings
    println("Tokenizer Configuration:")
    println("Normalizer type: ", config.normalizer.type)
    println("Pre-tokenizer settings:")
    println("  type: ", config.pre_tokenizer.type)
    println("  add_prefix_space: ", config.pre_tokenizer.add_prefix_space)
    println("  trim_offsets: ", config.pre_tokenizer.trim_offsets)
    println("  use_regex: ", config.pre_tokenizer.use_regex)
    
    # Analyze vocabulary
    vocab_size = length(config.model.vocab)
    println("\nVocabulary size: ", vocab_size)
    println("First 5 vocabulary entries:")
    for (i, (token, id)) in enumerate(config.model.vocab)
        i > 5 && break
        println("  $token => $id")
    end
    
    # Analyze merge rules
    merge_count = length(config.model.merges)
    println("\nMerge rules count: ", merge_count)
    println("First 5 merge rules:")
    for (i, merge_rule) in enumerate(config.model.merges)
        i > 5 && break
        println("  $merge_rule")
    end
    
    # Convert first few merges to BytePairEncoding format
    println("\nConverted merge format (first 5):")
    for (i, merge_rule) in enumerate(config.model.merges)
        i > 5 && break
        parts = split(String(merge_rule))
        if length(parts) == 2
            merge_pair = (Merge(parts[1]), Merge(parts[2]))
            println("  $merge_pair => $i")
        end
    end
    
    # Special tokens
    println("\nSpecial tokens:")
    for token in config.added_tokens
        if token.special
            println("  $(token.content) => $(token.id)")
        end
    end
end

# Run analysis
analyze_tokenizer_config()
