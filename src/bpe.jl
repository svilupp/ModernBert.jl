module BPE

export BPETokenizer, load_tokenizer, tokenize, bpe_encode, get_pairs, get_token_id, create_bpe_tokenizer

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
mutable struct BPETokenizer
    vocab::Dict{String,Int}
    merges::Vector{Tuple{String,String}}
    special_tokens::Dict{String,Int}
    special_token_properties::Dict{String,NamedTuple}
    cache::Dict{String,Vector{String}}
    prev_token::String
    
    function BPETokenizer(vocab::Dict{String,Int}, merges::Vector{Tuple{String,String}}, special_tokens::Dict{String,Int}, special_token_properties::Dict{String,NamedTuple}, cache::Dict{String,Vector{String}})
        new(vocab, merges, special_tokens, special_token_properties, cache, "[CLS]")
    end
end

"""
    load_tokenizer(config_path::String)

Load a BPE tokenizer configuration from a JSON file.

# Arguments
- `config_path::String`: Path to the tokenizer.json configuration file
"""
function load_tokenizer(config_path::String)
    config = JSON3.read(read(config_path, String))
    
    # Load vocabulary and merges from model configuration
    vocab = Dict{String,Int}()
    if haskey(config, "model") && haskey(config.model, "vocab")
        for (token, id) in config.model.vocab
            vocab[String(token)] = id  # IDs are already integers in the config
        end
    else
        error("No vocabulary found in tokenizer config model")
    end
    
    # Load merge rules preserving the exact order from config
    merges = Vector{Tuple{String,String}}()
    if haskey(config, "model") && haskey(config.model, "merges")
        for merge_rule in config.model.merges
            parts = split(String(merge_rule))
            if length(parts) == 2
                # Convert parts to strings and preserve the exact order
                p1, p2 = String(parts[1]), String(parts[2])
                push!(merges, (p1, p2))
            end
        end
    else
        error("No merge rules found in tokenizer config model")
    end
    
    # Load special tokens and their properties
    special_tokens = Dict{String,Int}()
    special_token_properties = Dict{String,NamedTuple}()
    
    # Load special tokens from config
    required_tokens = Dict{String,Int}()
    
    for token in config.added_tokens
        if token.special
            content = String(token.content)
            id = Int(token.id)
            
            # Store token ID from config without verification against hard-coded values
            special_tokens[content] = id
            special_token_properties[content] = (
                lstrip = Bool(token.lstrip),
                rstrip = Bool(token.rstrip),
                normalized = Bool(token.normalized),
                single_word = Bool(token.single_word)
            )
        end
    end
    
    # Load all special tokens from config into vocabulary
    for (token, id) in special_tokens
        vocab[token] = id
    end
    
    return BPETokenizer(vocab, merges, special_tokens, special_token_properties, Dict{String,Vector{String}}())
end

"""
    get_pairs(word::Vector{String})

Get all possible pairs and longer sequences in a word represented as tokens.
Returns pairs that could potentially be merged based on the vocabulary.
"""
function get_pairs(word::Vector{String})
    pairs = Set{Tuple{String,String}}()
    
    # Get all consecutive pairs and potential longer sequences
    for i in 1:length(word)-1
        # Basic pair
        push!(pairs, (word[i], word[i+1]))
        
        # Try longer sequences
        if i < length(word)-1
            # Try joining current token with next token
            seq = join([word[i], word[i+1]])
            if haskey(tokenizer.vocab, seq)
                # Add pair with next token
                push!(pairs, (seq, word[i+2]))
            end
            
            # Try joining next two tokens
            seq = join([word[i+1], word[i+2]])
            if haskey(tokenizer.vocab, seq)
                # Add pair with current token
                push!(pairs, (word[i], seq))
            end
        end
    end
    
    return pairs
end

"""
    tokenize(tokenizer::BPETokenizer, text::AbstractString; token_ids::Bool=false) -> Vector{Union{String,Int}}

Tokenize the input text using BPE tokenization.

# Arguments
- `text::AbstractString`: Text to tokenize (String or SubString)
- `token_ids::Bool=false`: Whether to return token IDs instead of strings

# Returns
- Vector of tokens (strings) or token IDs (integers)
"""
function tokenize(tokenizer::BPETokenizer, text::AbstractString; token_ids::Bool=false)
    # Convert SubString to String if necessary
    text_str = String(text)
    # Initialize result array with correct type
    tokens = token_ids ? Int[] : String[]
    
    # Handle empty input
    if isempty(text_str)
        return token_ids ? Int[] : String[]
    end
    
    # Split text into words and handle special characters
    words = String[]
    current_word = ""
    add_prefix = true  # First word gets prefix
    last_was_space = true  # Track if last character was a space
    
    i = firstindex(text_str)
    while i <= lastindex(text_str)
        char = text_str[i]
        if isspace(char)
            if !isempty(current_word)
                push!(words, current_word)
                current_word = ""
            end
            last_was_space = true
        elseif ispunct(char)
            # Handle punctuation as separate tokens
            if !isempty(current_word)
                push!(words, current_word)
                current_word = ""
            end
            push!(words, string(char))
            last_was_space = false  # Punctuation doesn't count as space
        else
            if isempty(current_word)
                # Start of a new word
                add_prefix = last_was_space  # Only add prefix if previous char was space
            end
            current_word *= string(char)
            last_was_space = false
        end
        
        # Handle last word
        if i == lastindex(text_str) && !isempty(current_word)
            push!(words, current_word)
        end
        
        i = nextind(text_str, i)
    end
    
    # Process each word
    for (i, word) in enumerate(words)
        # Skip empty words
        isempty(word) && continue
        
        # Check if word is a special token
        if haskey(tokenizer.special_tokens, word)
            # Update previous token state before adding current token
            tokenizer.prev_token = word
            
            # Add the token (as ID or string based on mode)
            if token_ids
                push!(tokens, tokenizer.special_tokens[word])
            else
                push!(tokens, word)
            end
            
            # Special tokens should be followed by a space unless it's punctuation
            last_was_space = !any(ispunct, word)
            add_prefix = true  # Words after special tokens should get prefix
            continue
        end
        
        # Determine if we should add prefix based on context
        # Add prefix if:
        # 1. Previous token was a space or special token
        # 2. Not the first word and not following punctuation
        should_prefix = if i == 1
            true  # First word gets prefix
        elseif i > 1 && haskey(tokenizer.special_tokens, words[i-1])
            true  # After special token gets prefix
        else
            add_prefix  # Use tracked state
        end
        
        # Apply BPE encoding with proper prefix handling and type conversion
        word_tokens = bpe_encode(tokenizer, word, should_prefix, token_ids)
        append!(tokens, word_tokens)
        
        # Update state for next word
        last_was_space = true  # Each word is followed by an implicit space
        add_prefix = true  # Next word should get prefix by default
    end
    
    # Convert to token IDs if requested
    if token_ids
        return [get(tokenizer.special_tokens, t, get(tokenizer.vocab, t, tokenizer.special_tokens["[UNK]"])) for t in tokens]
    end
    
    return tokens
end

"""
    bpe_encode(tokenizer::BPETokenizer, word::AbstractString, add_prefix::Bool=true)

Encode a single word using BPE merge operations.

# Arguments
- `word::AbstractString`: Word to encode (String or SubString)
- `add_prefix::Bool`: Whether to add the 'Ġ' prefix (for word boundaries)
"""
function bpe_encode(tokenizer::BPETokenizer, word::AbstractString, add_prefix::Bool=true, token_ids::Bool=false)
    # Convert SubString to String if necessary
    word_str = String(word)
    # Initialize arrays for token processing with correct type
    result = token_ids ? Int[] : String[]
    final_tokens = token_ids ? Int[] : String[]
    
    # Handle empty input
    isempty(word_str) && return result
    
    # Special handling for sentence-initial capitalized words
    is_sentence_initial = tokenizer.prev_token == "[CLS]" && !isempty(word_str) && 
        length(word_str) > 0 && isuppercase(first(word_str))
    
    # Check cache first
    cache_key = if is_sentence_initial
        word_str  # Use unprefixed version for sentence-initial words
    else
        add_prefix ? "Ġ" * word_str : word_str
    end
    
    if haskey(tokenizer.cache, cache_key)
        cached_result = tokenizer.cache[cache_key]
        if token_ids
            # Convert cached tokens to IDs
            return [get(tokenizer.vocab, t, tokenizer.special_tokens["[UNK]"]) for t in cached_result]
        end
        return copy(cached_result)
    end
    
    # Handle empty string or whitespace using vocabulary
    if isempty(word) || all(isspace, word)
        return token_ids ? Int[] : String[]
    end
    
    # Handle prefix logic
    # Don't add prefix if word already starts with Ġ or if it's sentence-initial
    has_prefix = startswith(word, "Ġ")
    actual_word = has_prefix ? SubString(word, nextind(word, 1)) : word
    should_add_prefix = add_prefix && !has_prefix && !is_sentence_initial
    
    # Handle special tokens with exact matching
    if haskey(tokenizer.special_tokens, word)
        token_id = tokenizer.special_tokens[word]
        return token_ids ? [token_id] : [word]
    end
    
    # For sentence-initial capitalized words
    if is_sentence_initial && !isempty(actual_word) && isuppercase(actual_word[1])
        # Try unprefixed version first for sentence-initial words
        if haskey(tokenizer.vocab, actual_word)
            token_id = tokenizer.vocab[actual_word]
            return token_ids ? [token_id] : [actual_word]
        end
    end
    
    # Handle complete word matches first with consistent prefix handling
    if !all(ispunct, actual_word) && !is_sentence_initial
        prefixed_word = "Ġ" * actual_word
        has_prefixed = haskey(tokenizer.vocab, prefixed_word)
        has_unprefixed = haskey(tokenizer.vocab, actual_word)
        
        # Determine if we're at a word boundary that requires a prefix
        # This includes:
        # 1. After a space (should_add_prefix is true)
        # 2. After a special token (check previous token)
        # 3. At the start of text (empty prev_token)
        is_after_special = haskey(tokenizer.special_tokens, tokenizer.prev_token)
        needs_prefix = should_add_prefix || is_after_special || tokenizer.prev_token == ""
        
        # Check for exact matches in vocabulary with proper prefix handling
        if needs_prefix
            # At word boundary, strongly prefer prefixed version
            if has_prefixed
                return token_ids ? [tokenizer.vocab[prefixed_word]] : [prefixed_word]
            elseif has_unprefixed
                # Fallback to unprefixed if that's all we have
                return token_ids ? [tokenizer.vocab[actual_word]] : [actual_word]
            end
        else
            # Not at word boundary, try unprefixed first
            if has_unprefixed
                return token_ids ? [tokenizer.vocab[actual_word]] : [actual_word]
            elseif has_prefixed
                return token_ids ? [tokenizer.vocab[prefixed_word]] : [prefixed_word]
            end
        end
    # For sentence-initial words, try unprefixed first
    elseif is_sentence_initial && haskey(tokenizer.vocab, actual_word)
        return token_ids ? [tokenizer.vocab[actual_word]] : [actual_word]
    # For punctuation, always use unprefixed
    elseif all(ispunct, actual_word) && haskey(tokenizer.vocab, actual_word)
        return token_ids ? [tokenizer.vocab[actual_word]] : [actual_word]
    end
    
    # Handle punctuation
    if length(word) == 1 && any(ispunct, word)
        token_id = get(tokenizer.vocab, word, tokenizer.special_tokens["[UNK]"])
        return token_ids ? [token_id] : [word]
    end

    # Initialize result array for BPE tokenization
    result = Vector{String}()
    
    # Handle empty words or single characters
    if isempty(actual_word)
        return token_ids ? Int[] : String[]
    end
    
    # Initialize with character-level tokens following BPE rules
    chars = String[]
    
    # Handle the prefix for the first character
    if should_add_prefix
        push!(chars, "Ġ")
    end
    
    # Add each character as a separate token
    for c in actual_word
        # Skip if it's a combining character or zero-width character
        if !isempty(string(c))
            push!(chars, string(c))
        end
    end
    
    # Initialize result with character tokens
    result = chars
    
    # Create a mapping of merge rule indices for faster lookup
    merge_indices = Dict{Tuple{String,String}, Int}()
    for (idx, pair) in enumerate(tokenizer.merges)
        merge_indices[pair] = idx
    end
    
    # Keep applying merge rules until no more merges are possible
    while true
        valid_merges = Tuple{Int,Int,Int}[]  # (start_idx, end_idx, merge_rule_idx)
        
        # Find all possible merges in current sequence
        for i in 1:length(result)-1
            pair = (result[i], result[i+1])
            if haskey(merge_indices, pair)
                push!(valid_merges, (i, i+1, merge_indices[pair]))
            end
        end
        
        # If no valid merges found, we're done
        isempty(valid_merges) && break
        
        # Apply the merge with lowest rule index (highest priority)
        sort!(valid_merges, by=x -> x[3])
        start_idx, end_idx, _ = valid_merges[1]
        merged = join(result[start_idx:end_idx])
        
        # Update result with merged token
        result = vcat(result[1:start_idx-1], [merged], result[end_idx+1:end])
    end
    
    # Apply merge rules from tokenizer.json with correct priority
    iteration_count = 0
    max_iterations = 1000  # Prevent infinite loops
    
    while iteration_count < max_iterations
        iteration_count += 1
        
        # Try longer sequences first
        valid_merges = Vector{Tuple{Vector{String}, Int, String}}()
        
        # Apply merge rules in order from tokenizer.json
        # Only look for basic pairs - no complex sequences
        for i in 1:length(result)-1
            pair = (result[i], result[i+1])
            # Skip pairs containing special tokens
            if any(t -> haskey(tokenizer.special_tokens, t), pair)
                continue
            end
            
            # Try to find this pair in merge rules
            merged = join(pair)
            if haskey(tokenizer.vocab, merged)
                idx = findfirst(==(pair), tokenizer.merges)
                if !isnothing(idx)
                    push!(valid_merges, ([pair[1], pair[2]], idx, merged))
                end
            end
            
            # Also try with Ġ prefix if appropriate
            if !startswith(merged, "Ġ") && should_add_prefix
                prefixed = "Ġ" * merged
                if haskey(tokenizer.vocab, prefixed)
                    idx = findfirst(==(pair), tokenizer.merges)
                    if !isnothing(idx)
                        push!(valid_merges, ([pair[1], pair[2]], idx, prefixed))
                    end
                end
            end
        end
        if isempty(valid_merges)
            for i in 1:length(result)-1
                pair = (result[i], result[i+1])
                # Skip pairs containing special tokens
                if any(t -> haskey(tokenizer.special_tokens, t), pair)
                    continue
                end
                
                idx = findfirst(==(pair), tokenizer.merges)
                if !isnothing(idx)
                    merged = join(pair)
                    if haskey(tokenizer.vocab, merged)
                        push!(valid_merges, ([pair[1], pair[2]], idx, merged))
                    elseif !startswith(merged, "Ġ") && haskey(tokenizer.vocab, "Ġ" * merged)
                        push!(valid_merges, ([pair[1], pair[2]], idx, "Ġ" * merged))
                    end
                end
            end
        end
        
        # Sort by merge rule index and apply best merge
        if !isempty(valid_merges)
            # Sort by merge rule index only - lower index means higher priority
            sort!(valid_merges, by=x -> x[2])
            best_pair = valid_merges[1][1]
            merged_token = valid_merges[1][3]  # Already validated in vocab
            
            # Apply the merge using the correct merged token and pair
            new_result = String[]
            i = 1
            while i <= length(result)
                if i < length(result) && result[i] == best_pair[1] && result[i+1] == best_pair[2]
                    push!(new_result, merged_token)
                    i += 2
                else
                    push!(new_result, result[i])
                    i += 1
                end
            end
            
            # Check if the merge actually changed anything
            if new_result == result
                break  # No changes made, stop merging
            end
            
            result = new_result
        else
            break  # No valid pairs to merge
        end
    end
    
    # Cache and return results
    
    # Only cache if we have valid tokens
    if !isempty(result) && all(t -> haskey(tokenizer.vocab, t) || haskey(tokenizer.special_tokens, t), result)
        tokenizer.cache[cache_key] = copy(result)
    end
    
    # Convert to token IDs if requested, otherwise return strings
    if token_ids
        # Convert tokens to IDs using vocabulary and special tokens
        token_ids = Int[]
        for token in result
            if haskey(tokenizer.special_tokens, token)
                push!(token_ids, tokenizer.special_tokens[token])
            else
                push!(token_ids, get(tokenizer.vocab, token, tokenizer.special_tokens["[UNK]"]))
            end
        end
        return token_ids
    end
    return result
end

# Helper function to get token ID
function get_token_id(tokenizer::BPETokenizer, token::String)
    return get(tokenizer.vocab, token, tokenizer.special_tokens["[UNK]"])
end

"""
    (tokenizer::BPETokenizer)(text::AbstractString; token_ids::Bool=false, add_special_tokens::Bool=true)

Function call overloading for BPETokenizer to make it callable directly.
Tokenizes text using the BPE algorithm.

# Arguments
- `text::AbstractString`: Input text to tokenize
- `token_ids::Bool=false`: If true, return token IDs instead of tokens
- `add_special_tokens::Bool=true`: Whether to add special tokens (ignored in BPE tokenizer)

# Returns
- Vector of tokens or token IDs
"""
function (tokenizer::BPETokenizer)(text::AbstractString; token_ids::Bool=false, add_special_tokens::Bool=true)
    # Handle empty input
    isempty(text) && return token_ids ? Int[] : String[]
    
    # Reset previous token state at the start of tokenization
    # Only initialize if we're not being called through the encoder
    # (encoder handles special tokens separately)
    tokenizer.prev_token = ""  # Always start with empty state to ensure consistent behavior
    
    # Initialize result vector with correct type based on token_ids parameter
    tokens = token_ids ? Vector{Int}() : Vector{String}()
    
    # Get list of special tokens for preservation
    special_token_list = collect(keys(tokenizer.special_tokens))
    
    # Split text into words while preserving whitespace and special tokens
    current_pos = 1
    text_length = length(text)
    last_was_space = true  # Start assuming we had a space (for first word)
    
    while current_pos <= text_length
        # Skip consecutive whitespace
        while current_pos <= text_length && isspace(text[current_pos])
            current_pos += 1
            last_was_space = true
        end
        current_pos > text_length && break
        
        # Check for special tokens first
        found_special = false
        for special_token in special_token_list
            if startswith(@view(text[current_pos:end]), special_token)
                # Handle special token properties
                if haskey(tokenizer.special_token_properties, special_token)
                    props = tokenizer.special_token_properties[special_token]
                    if props.lstrip
                        last_was_space = true  # Force space before next token
                    end
                    if props.rstrip
                        last_was_space = true  # Force space after this token
                    end
                end
                
                # Update previous token before adding current token
                # This ensures proper handling of the next word
                # Always store the string representation of the token
                tokenizer.prev_token = special_token
                
                if token_ids
                    # Convert special token to ID directly
                    token_id = tokenizer.special_tokens[special_token]
                    push!(tokens, token_id)
                else
                    push!(tokens, special_token)
                end
                
                current_pos += length(special_token)
                found_special = true
                # Force space after special tokens to ensure proper word boundary
                last_was_space = true
                break
            end
        end
        
        # Skip word processing if we found a special token
        if found_special
            continue
        end
        
        # Find word boundary
        word_start = current_pos
        while current_pos <= text_length
            if isspace(text[current_pos]) || any(special -> startswith(@view(text[current_pos:end]), special), special_token_list)
                break
            end
            current_pos += 1
        end
        
        # Extract and process word
        if current_pos > word_start
            word = text[word_start:current_pos-1]
            # Special handling for sentence-initial capitalized words
            is_sentence_initial = tokenizer.prev_token == "[CLS]" && length(word) > 0 && isuppercase(word[1])
            
            if is_sentence_initial && haskey(tokenizer.vocab, word)
                # For sentence-initial capitalized words, prefer unprefixed version
                word_tokens = token_ids ? [tokenizer.vocab[word]] : [word]
            else
                # For all other cases, try complete word matches first
                prefixed_word = "Ġ" * word
                if !all(ispunct, word)  # Don't prefix punctuation
                    # Check if we're after a special token
                    is_after_special = haskey(tokenizer.special_tokens, tokenizer.prev_token)
                    
                    # Always check both prefixed and unprefixed versions
                    has_prefixed = haskey(tokenizer.vocab, prefixed_word)
                    has_unprefixed = haskey(tokenizer.vocab, word)
                    
                    # First, try exact matches with proper prefix handling
                    if has_prefixed || has_unprefixed
                        # After special tokens or at start of text, ALWAYS use prefixed version if available
                        if (is_after_special || tokenizer.prev_token == "") && has_prefixed
                            word_tokens = token_ids ? [tokenizer.vocab[prefixed_word]] : [prefixed_word]
                        # For other cases, check if we have a complete word match
                        elseif has_unprefixed && !is_after_special
                            # Use unprefixed version for complete words in middle of text
                            word_tokens = token_ids ? [tokenizer.vocab[word]] : [word]
                        elseif has_prefixed
                            # Fallback to prefixed version if available
                            word_tokens = token_ids ? [tokenizer.vocab[prefixed_word]] : [prefixed_word]
                        else
                            # Last resort: use unprefixed version
                            word_tokens = token_ids ? [tokenizer.vocab[word]] : [word]
                        end
                    else
                        # Only if no complete match exists, try BPE encoding
                        # Add prefix at word boundaries (after special tokens or start)
                        # Check for complete word match first
                        prefixed_word = "Ġ" * word
                        if haskey(tokenizer.vocab, prefixed_word) && (is_after_special || tokenizer.prev_token == "")
                            word_tokens = token_ids ? [tokenizer.vocab[prefixed_word]] : [prefixed_word]
                        elseif haskey(tokenizer.vocab, word)
                            word_tokens = token_ids ? [tokenizer.vocab[word]] : [word]
                        else
                            # Only do BPE if no complete match exists
                            word_tokens = bpe_encode(tokenizer, word, is_after_special || tokenizer.prev_token == "", token_ids)
                        end
                    end
                else
                    # Handle punctuation without prefix
                    if haskey(tokenizer.vocab, word)
                        word_tokens = token_ids ? [tokenizer.vocab[word]] : [word]
                    else
                        word_tokens = bpe_encode(tokenizer, word, false, token_ids)
                    end
                end
            end
            append!(tokens, word_tokens)
            # Update previous token and handle word boundaries
            if !isempty(word_tokens)
                # For token_ids mode, we need to find the original token string
                if token_ids
                    # Try to find the token string from special tokens first
                    token_str = nothing
                    token_id = word_tokens[end]
                    
                    # Check special tokens first
                    for (special_token, id) in tokenizer.special_tokens
                        if id == token_id
                            token_str = special_token
                            break
                        end
                    end
                    
                    # If not a special token, look in regular vocab
                    if isnothing(token_str)
                        for (token, id) in tokenizer.vocab
                            if id == token_id
                                token_str = token
                                break
                            end
                        end
                    end
                    
                    # Update prev_token with the found string or UNK
                    tokenizer.prev_token = something(token_str, "[UNK]")
                else
                    # In string mode, just use the token directly
                    tokenizer.prev_token = word_tokens[end]
                end
            end
            last_was_space = true  # Ensure proper word boundary after each word
        end
    end
    
    # Return tokens (already in correct type)
    return tokens
end

# Convenience method to create a BPE tokenizer from a config file
function create_bpe_tokenizer(config_path::String)
    if !isfile(config_path)
        error("Tokenizer configuration file not found: $config_path")
    end
    return load_tokenizer(config_path)
end

end # module BPE
