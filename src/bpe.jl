## This implements a Byte-Pair Encoding (BPE) tokenizer for ModernBERT
using JSON3

"""
    BPETokenizer

A BPE (Byte-Pair Encoding) tokenizer that splits text into subword units based on learned merge rules.

# Fields
- `vocab::Dict{String,Int}`: Mapping from tokens to their IDs (0-based for Python compatibility)
- `merges::Vector{Tuple{String,String}}`: Ordered list of BPE merge rules
- `special_tokens::Dict{String,Int}`: Special token mapping (e.g., [CLS], [SEP], etc.)
- `special_token_properties::Dict{String,NamedTuple}`: Properties for special tokens (lstrip, rstrip, etc.)
- `cache::Dict{String,Vector{String}}`: Cache for already tokenized words
"""
struct BPETokenizer
    vocab::Dict{String,Int}
    merges::Vector{Tuple{String,String}}
    special_tokens::Dict{String,Int}
    special_token_properties::Dict{String,NamedTuple}
    cache::Dict{String,Vector{String}}
end

"""
    load_tokenizer(config_path::String)

Load a BPE tokenizer configuration from a JSON file.

# Arguments
- `config_path::String`: Path to the tokenizer.json configuration file
"""
function load_tokenizer(config_path::String)
    config = JSON3.read(read(config_path, String))
    
    # Load vocabulary and merges
    vocab = Dict{String,Int}()
    for (token, id) in config.model.vocab
        vocab[String(token)] = id
    end
    
    # Load merge rules
    merges = Vector{Tuple{String,String}}()
    for merge_rule in config.model.merges
        p1, p2 = split(String(merge_rule))
        push!(merges, (String(p1), String(p2)))
    end
    
    # Load special tokens and their properties
    special_tokens = Dict{String,Int}()
    special_token_properties = Dict{String,NamedTuple}()
    
    # Ensure required special tokens are present with correct IDs
    required_tokens = Dict{String,Int}(
        "[CLS]" => 50281,
        "[SEP]" => 50282,
        "[MASK]" => 50284,
        "[PAD]" => 50283,
        "[UNK]" => 50280,
        "<|padding|>" => 1,
        "<|endoftext|>" => 50279
    )
    
    for token in config.added_tokens
        if token.special
            content = String(token.content)
            id = Int(token.id)
            
            # Verify ID matches expected value for required tokens
            if haskey(required_tokens, content)
                @assert id == required_tokens[content] "Mismatched ID for token $content: expected $(required_tokens[content]), got $id"
            end
            
            special_tokens[content] = id
            special_token_properties[content] = (
                lstrip = Bool(token.lstrip),
                rstrip = Bool(token.rstrip),
                normalized = Bool(token.normalized),
                single_word = Bool(token.single_word)
            )
        end
    end
    
    # Verify all required tokens are present
    for (token, expected_id) in required_tokens
        if !haskey(special_tokens, token)
            error("Required special token $token (ID: $expected_id) not found in tokenizer config")
        end
    end
    
    return BPETokenizer(vocab, merges, special_tokens, special_token_properties, Dict{String,Vector{String}}())
end

"""
    get_pairs(word::Vector{String})

Get all bigram pairs in a word represented as a sequence of tokens.
"""
function get_pairs(word::Vector{String})
    pairs = Set{Tuple{String,String}}()
    for i in 1:length(word)-1
        push!(pairs, (word[i], word[i+1]))
    end
    return pairs
end

"""
    bpe_encode(tokenizer::BPETokenizer, word::String, add_prefix::Bool=true)

Encode a single word using BPE merge operations.

# Arguments
- `word::String`: Word to encode
- `add_prefix::Bool`: Whether to add the 'Ġ' prefix (for word boundaries)
"""
function bpe_encode(tokenizer::BPETokenizer, word::String, add_prefix::Bool=true)
    # Check cache first
    cache_key = add_prefix ? "Ġ" * word : word
    haskey(tokenizer.cache, cache_key) && return tokenizer.cache[cache_key]
    
    # Handle empty string or whitespace
    if isempty(word) || all(isspace, word)
        spaces = count(isequal(' '), word)
        if spaces == 0
            return String[]
        elseif spaces == 1
            return ["Ġ"]
        elseif spaces == 2
            return ["ĠĠ"]
        else
            return ["Ġ"^spaces]
        end
    end
    
    # Add 'Ġ' prefix for word boundaries if needed
    word_with_prefix = add_prefix ? "Ġ" * word : word
    
    # Convert word to sequence of characters
    chars = string.(collect(word_with_prefix))
    
    while true
        pairs = get_pairs(chars)
        isempty(pairs) && break
        
        # Find the highest priority pair that exists in our merges
        best_pair = nothing
        for pair in tokenizer.merges
            if pair in pairs
                best_pair = pair
                break
            end
        end
        
        isnothing(best_pair) && break
        
        # Merge the pair throughout the word
        new_chars = Vector{String}()
        i = 1
        while i <= length(chars)
            if i < length(chars) && chars[i] == best_pair[1] && chars[i+1] == best_pair[2]
                push!(new_chars, chars[i] * chars[i+1])
                i += 2
            else
                push!(new_chars, chars[i])
                i += 1
            end
        end
        chars = new_chars
    end
    
    # Filter out empty tokens and cache the result
    result = filter(!isempty, chars)
    tokenizer.cache[cache_key] = result
    return result
end

"""
    (tokenizer::BPETokenizer)(text::AbstractString; token_ids::Bool=false)

Function call overloading for BPETokenizer to make it callable directly.
Tokenizes text using the BPE algorithm.

# Arguments
- `text::AbstractString`: Input text to tokenize
- `token_ids::Bool=false`: If true, return token IDs instead of tokens

# Returns
- Vector of tokens or token IDs
"""
function (tokenizer::BPETokenizer)(text::AbstractString; token_ids::Bool=false)
    # Handle empty input
    isempty(text) && return token_ids ? Int[] : String[]
    
    # Get list of special tokens for preservation during basic tokenization
    special_token_list = collect(keys(tokenizer.special_tokens))
    
    # Basic tokenization with special token preservation
    basic_tokens = _bert_tokenise(text, Val(false), special_token_list)  # Use cased version as BPE handles casing
    
    # Apply BPE to each token
    tokens = String[]
    for (i, token) in enumerate(basic_tokens)
        # Check if it's a special token first
        if haskey(tokenizer.special_tokens, token)
            # Handle special token properties
            props = tokenizer.special_token_properties[token]
            
            # Apply lstrip/rstrip if needed
            if props.lstrip && i > 1 && !isempty(tokens)
                # Remove leading space from current token for lstrip
                token = lstrip(token)
            end
            if props.rstrip && i < length(basic_tokens)
                # Remove trailing space for rstrip
                token = rstrip(token)
            end
            
            push!(tokens, token)
            continue
        end
        
        # Handle pure whitespace tokens
        if all(isspace, token)
            append!(tokens, bpe_encode(tokenizer, token))
            continue
        end
        
        # Apply BPE encoding with 'Ġ' prefix for word boundaries
        # Don't add prefix for the first token if it's at the start of the text
        # or if the previous token wasn't whitespace
        # Also don't add prefix for punctuation
        add_prefix = i > 1 && !all(isspace, basic_tokens[i-1]) && !isbertpunct(first(token))
        bpe_tokens = bpe_encode(tokenizer, token, add_prefix)
        append!(tokens, bpe_tokens)
    end
    
    # Convert to token IDs if requested
    if token_ids
        return [get(tokenizer.special_tokens, t, get(tokenizer.vocab, t, tokenizer.special_tokens["[UNK]"])) for t in tokens]
    end
    
    return tokens
end

# Convenience method to create a BPE tokenizer from a config file
function create_bpe_tokenizer(config_path::String)
    if !isfile(config_path)
        error("Tokenizer configuration file not found: $config_path")
    end
    return load_tokenizer(config_path)
end
