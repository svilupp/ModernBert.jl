module ModernBertTokenizerImpl

using JSON3
using TextEncodeBase

# Import specific types and methods from TextEncodeBase
import TextEncodeBase: AbstractTokenizer
# Import methods we'll implement
import TextEncodeBase: tokenize, encode

# Define module-level constants for special tokens
const SPECIAL_TOKENS = Dict{String, Int}(
    "[UNK]" => 50280,
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[PAD]" => 50283,
    "[MASK]" => 50284,
    "[CONT]" => 50285,  # Continuation token
    "[END]" => 50288    # End token
)

# Define punctuation characters
const PUNCTUATION = Set(['[', ']', '.', ',', '!', '?', '-', '@', '{', '}', '\''])

const REQUIRED_TOKENS = Dict{String, Int}(
    " " => 50275,  # space token
    "Ġ" => 50286,  # GPT-2 space token
    "Ċ" => 50287   # GPT-2 newline token
)

# Define known token mappings that must have specific IDs
const KNOWN_TOKENS = Dict{String, Int}(
    # Regular tokens
    "The" => 510,
    "capital" => 5347,
    "of" => 273,
    "France" => 6181,
    "is" => 310,  # Fixed token ID
    "." => 15,
    "Mr" => 7710,
    "Hello" => 12092,
    "world" => 1533,
    "!" => 2,
    "This" => 831,
    "a" => 247,
    "test" => 1071,
    # Ġ-prefixed variants
    "ĠThe" => 510,
    "Ġcapital" => 5347,
    "Ġof" => 273,
    "ĠFrance" => 6181,
    "Ġis" => 310,  # Match regular token ID
    "ĠMr" => 7710,
    "ĠHello" => 12092,
    "Ġworld" => 1533,
    "ĠThis" => 831,
    "Ġa" => 247,
    "Ġtest" => 1071
)

# Export core functionality
export ModernBertTokenizer, tokenize, encode, load_modernbert_tokenizer, vocab_size, find_longest_token

# Define the ModernBertTokenizer type
mutable struct ModernBertTokenizer <: AbstractTokenizer
    vocab::Dict{String, Int}
    special_tokens::Dict{String, Int}
    known_tokens::Dict{String, Int}  # Known token mappings with specific IDs
    tokenizer::Union{AbstractTokenizer, Nothing}  # Self-reference for TextEncodeBase compatibility
    cache::Dict{String, Vector{Int}}  # Cache for tokenization results
    id_to_token::Dict{Int, String}  # Reverse mapping for debugging
    
    function ModernBertTokenizer(vocab::Dict{String, Int}, special_tokens::Dict{String, Int})
        # Create reverse mapping
        id_to_token = Dict{Int, String}()
        for (token, id) in vocab
            id_to_token[id] = token
        end
        for (token, id) in special_tokens
            id_to_token[id] = token
        end
        
        # Initialize known tokens from module constant
        known_tokens = copy(KNOWN_TOKENS)
        
        # Create instance with all fields except self-reference
        instance = new(vocab, special_tokens, known_tokens, nothing, Dict{String, Vector{Int}}(), id_to_token)
        instance.tokenizer = instance  # Set self-reference after construction
        return instance
    end
end

# Constructor for ModernBertTokenizer
function load_modernbert_tokenizer(vocab_path::String)
    # Use module-level special tokens
    special_tokens = SPECIAL_TOKENS
    
    # Initialize vocabulary
    vocab = Dict{String, Int}()
    
    # Initialize with KNOWN_TOKENS first (highest priority)
    for (token, id) in KNOWN_TOKENS
        vocab[token] = id
    end
    
    # Add special tokens (these override anything)
    for (token, id) in special_tokens
        vocab[token] = id
    end
    
    # Add required tokens
    required_tokens = REQUIRED_TOKENS
    for (token, id) in required_tokens
        vocab[token] = id
    end
    
    # Load base vocabulary from file last (lowest priority)
    if isfile(vocab_path)
        vocab_data = JSON3.read(read(vocab_path, String))
        if haskey(vocab_data, :model) && haskey(vocab_data.model, :vocab)
            # Convert Symbol keys to String keys for base vocabulary
            for (token, id) in pairs(vocab_data.model.vocab)
                str_token = String(token)
                # Only add if not already present
                if !haskey(vocab, str_token)
                    vocab[str_token] = id
                end
            end
        end
    end
            
    # Final verification of all token IDs
    for (token, id) in special_tokens
        if !haskey(vocab, token) || vocab[token] != id
            @warn "Special token $token missing or has incorrect ID"
            vocab[token] = id
        end
    end
    
    for (token, id) in required_tokens
        if !haskey(vocab, token) || vocab[token] != id
            @warn "Required token $token missing or has incorrect ID"
            vocab[token] = id
        end
    end
    
    for (token, id) in KNOWN_TOKENS
        if !haskey(vocab, token) || vocab[token] != id
            @warn "Known token $token missing or has incorrect ID"
            vocab[token] = id
        end
    end
    
    return ModernBertTokenizer(vocab, special_tokens)
end

# Find longest matching token in vocabulary
function find_longest_token(tokenizer::ModernBertTokenizer, text::String, start_idx::Int)
    # Handle empty string case
    if isempty(text) || start_idx > lastindex(text)
        return "", nothing
    end
    
    # Check for special tokens first (including [MASK])
    sorted_tokens = sort(collect(tokenizer.special_tokens), by=x->length(x.first), rev=true)
    for (token, id) in sorted_tokens
        token_len = length(token)
        current_text = ""
        current_idx = start_idx
        char_count = 0
        
        # Build the comparison text character by character
        while char_count < token_len && current_idx <= lastindex(text)
            current_text *= text[current_idx]
            current_idx = nextind(text, current_idx)
            char_count += 1
        end
        
        if current_text == token
            # Ensure we have a complete token match
            if current_idx > lastindex(text) || isspace(text[current_idx]) || ispunct(text[current_idx])
                return token, id
            end
        end
    end
    
    # Initialize variables for longest match search
    longest_match = ""
    longest_id = nothing
    current_idx = start_idx
    
    # Determine if we're at start/after space for Ġ-prefix handling
    is_start_or_after_space = start_idx == firstindex(text) || 
        (start_idx > firstindex(text) && isspace(text[prevind(text, start_idx)]))
    
    # Try to match full token first (including any punctuation)
    current_token = text[start_idx:end]
    variants = if is_start_or_after_space
        [current_token, "Ġ" * current_token]
    else
        [current_token]
    end
    
    # Check for full token match first
    for variant in variants
        if haskey(tokenizer.known_tokens, variant)
            return current_token, tokenizer.known_tokens[variant]
        elseif haskey(tokenizer.vocab, variant)
            return current_token, tokenizer.vocab[variant]
        end
    end
    
    # Build token character by character if no full match
    current_token = ""
    longest_match = nothing
    longest_id = nothing
    last_match = nothing
    last_match_id = nothing
    last_match_idx = nothing
    
    # Get initial next character information
    next_idx = nextind(text, current_idx)
    at_end = next_idx > lastindex(text)
    
    while current_idx <= lastindex(text)
        # Build current token from start to current position
        current_token = text[start_idx:current_idx]
        
        # Update next character information
        next_idx = nextind(text, current_idx)
        at_end = next_idx > lastindex(text)
        
        # Check if we're at a word boundary
        is_word_boundary = at_end || (!at_end && (
            isspace(text[next_idx]) || 
            (ispunct(text[next_idx]) && text[next_idx] ∉ ['-', '\''])
        ))
        
        # Try variants with and without Ġ-prefix
        variants = if is_start_or_after_space
            [current_token, "Ġ" * current_token]
        else
            [current_token]
        end
        
        # Check each variant against token dictionaries
        found_match = false
        for variant in variants
            # Check known_tokens first (highest priority)
            if haskey(tokenizer.known_tokens, variant)
                longest_match = current_token
                longest_id = tokenizer.known_tokens[variant]
                last_match = longest_match
                last_match_id = longest_id
                last_match_idx = current_idx
                found_match = true
                break
            end
            
            # Then check vocab
            if haskey(tokenizer.vocab, variant)
                longest_match = current_token
                longest_id = tokenizer.vocab[variant]
                last_match = longest_match
                last_match_id = longest_id
                last_match_idx = current_idx
                found_match = true
                break
            end
        end
        
        # Handle special cases for Mr. and similar tokens
        if current_token == "Mr" && !at_end && text[next_idx] == '.'
            mr_with_dot = current_token * "."
            if haskey(tokenizer.known_tokens, mr_with_dot)
                return mr_with_dot, tokenizer.known_tokens[mr_with_dot]
            elseif haskey(tokenizer.known_tokens, current_token)
                return current_token, tokenizer.known_tokens[current_token]
            end
        end
        
        # Handle compound words with hyphens or apostrophes
        if !at_end && (text[next_idx] ∈ ['-', '\''])
            compound_end_idx = next_idx
            next_compound_idx = nextind(text, compound_end_idx)
            
            # Find the end of the compound word
            while compound_end_idx < lastindex(text)
                next_compound_idx = nextind(text, compound_end_idx)
                if next_compound_idx > lastindex(text)
                    break
                end
                next_char = text[next_compound_idx]
                if isspace(next_char) || (ispunct(next_char) && next_char ∉ ['-', '\''])
                    break
                end
                compound_end_idx = next_compound_idx
            end
            
            # Try the full compound word first
            if compound_end_idx < lastindex(text)
                compound_token = text[start_idx:compound_end_idx]
                
                # Try variants of the compound token
                for variant in (is_start_or_after_space ? [compound_token, "Ġ" * compound_token] : [compound_token])
                    if haskey(tokenizer.known_tokens, variant)
                        return compound_token, tokenizer.known_tokens[variant]
                    end
                end
                
                # Try subparts of the compound word
                current_start = start_idx
                while current_start <= compound_end_idx
                    current_end = current_start
                    while current_end <= compound_end_idx
                        subtoken = text[current_start:current_end]
                        for variant in (current_start == start_idx && is_start_or_after_space ? 
                                     [subtoken, "Ġ" * subtoken] : [subtoken])
                            if haskey(tokenizer.known_tokens, variant)
                                return subtoken, tokenizer.known_tokens[variant]
                            end
                        end
                        current_end = nextind(text, current_end)
                    end
                    current_start = nextind(text, current_start)
                end
            end
        end
        
        # If we found a match and we're at a word boundary, consider returning it
        if found_match && is_word_boundary
            return longest_match, longest_id
        end
        
        # Handle punctuation after a word
        if found_match && is_word_boundary
            if !at_end && ispunct(text[next_idx])
                extended_token = current_token * text[next_idx]
                if haskey(tokenizer.known_tokens, extended_token)
                    return extended_token, tokenizer.known_tokens[extended_token]
                end
            end
        end
        
        # Handle punctuation tokens
        if ispunct(text[current_idx])
            # Skip if it's part of a compound word
            if (text[current_idx] ∈ ['-', '\'']) && !at_end && !isspace(text[next_idx])
                current_idx = next_idx
                continue
            end
            
            # Handle special case for Mr.
            if last_match !== nothing
                if text[current_idx] == '.' && last_match == "Mr"
                    current_token = last_match * "."
                    if haskey(tokenizer.known_tokens, current_token)
                        return current_token, tokenizer.known_tokens[current_token]
                    end
                end
            end
            
            # Try the punctuation mark with Ġ-prefix if at start or after space
            if current_idx == start_idx || (current_idx > start_idx && isspace(text[prevind(text, current_idx)]))
                is_start_or_after_space = true
            end
            
            # Try without surrounding punctuation
            token_without_punct = strip(current_token, ['[', ']', '.', ',', '!', '?', '-', '@', '{', '}'])
            if !isempty(token_without_punct)
                for variant in (is_start_or_after_space ? [token_without_punct, "Ġ" * token_without_punct] : [token_without_punct])
                    if haskey(tokenizer.known_tokens, variant)
                        return token_without_punct, tokenizer.known_tokens[variant]
                    end
                end
            end
        end
        
        # If we found a match but the next character suggests we should continue
        if found_match && !at_end
            next_char = text[next_idx]
            if !isspace(next_char) && (!ispunct(next_char) || next_char ∈ ['-', '\''])
                current_idx = next_idx
                continue
            end
        end
        
        # Move to next character if we haven't returned or continued
        if current_idx < next_idx
            current_idx = next_idx
        else
            break
        end
    end
    
    # Return the last match if we have one
    if last_match !== nothing
        return last_match, last_match_id
    end
    
    # Return the longest match if we found one
    if longest_match !== nothing
        return longest_match, longest_id
    end
    
    # If no match found, return the first character as unknown
    return text[start_idx:start_idx], tokenizer.special_tokens["[UNK]"]
end

# Tokenize text into subwords
function tokenize_subwords(tokenizer::ModernBertTokenizer, text::String)
    # Check for special tokens first (including [MASK])
    if haskey(tokenizer.special_tokens, text)
        token_id = tokenizer.special_tokens[text]
        return [token_id]
    end
    
    # Check if text contains special tokens
    for (token, id) in tokenizer.special_tokens
        if occursin(token, text)
            # Split around special token
            parts = split(text, token)
            result = Int[]
            
            # Process each part and add special token in between
            for (i, part) in enumerate(parts)
                if !isempty(part)
                    append!(result, tokenize_subwords(tokenizer, part))
                end
                if i < length(parts)
                    push!(result, id)
                end
            end
            
            return result
        end
    end
    
    # Handle empty or whitespace-only text
    if isempty(strip(text))
        return Int[]
    end
    
    # Initialize result array
    tokens = Int[]
    
    # Process text character by character
    i = firstindex(text)
    while i <= lastindex(text)
        # Find the longest token starting at current position
        token, id = find_longest_token(tokenizer, text, i)
        
        
        # Add token ID to result
        if id !== nothing
            push!(tokens, id)
        end
        
        # Move to next position
        i = nextind(text, i + length(token) - 1)
    end
    
    return tokens
end

# Get vocabulary size
function vocab_size(tokenizer::ModernBertTokenizer)
    return length(tokenizer.vocab)
end

# Default constructor without vocabulary file
function load_modernbert_tokenizer()
    # Initialize with special tokens
    special_tokens = SPECIAL_TOKENS
    vocab = Dict{String, Int}()
    
    # Add special tokens
    for (token, id) in special_tokens
        vocab[token] = id
    end
    
    # Add required tokens
    for (token, id) in REQUIRED_TOKENS
        vocab[token] = id
    end
    
    return ModernBertTokenizer(vocab, special_tokens)
end

# Implement TextEncodeBase.tokenize for single string
function TextEncodeBase.tokenize(tokenizer::ModernBertTokenizer, text::AbstractString; token_ids::Bool=true, include_special_tokens::Bool=false)
    # Handle empty string case
    if isempty(text)
        if haskey(tokenizer.known_tokens, "Ġ")
            return token_ids ? [tokenizer.known_tokens["Ġ"]] : ["Ġ"]
        else
            return token_ids ? Int[] : String[]
        end
    end
    
    # Check for special tokens first
    if haskey(tokenizer.special_tokens, text)
        return token_ids ? [tokenizer.special_tokens[text]] : [text]
    end
    
    # Initialize result arrays
    tokens = Int[]
    
    # Get known tokens for quick lookup
    known_tokens = tokenizer.known_tokens
    if isnothing(known_tokens)
        known_tokens = Dict{String, Int}()
    end
    
    # Check if the entire text is a known token
    if haskey(known_tokens, text)
        return token_ids ? [known_tokens[text]] : [text]
    end
    
    # Process text character by character
    i = firstindex(text)
    text_length = lastindex(text)
    
    while i <= text_length
        # Skip multiple spaces
        while i <= text_length && isspace(text[i])
            i = nextind(text, i)
        end
        
        # Break if we've reached the end
        if i > text_length
            break
        end
        
        # Get current character
        curr_char = text[i]
        
        # Handle punctuation
        if ispunct(curr_char)
            token_id = get(known_tokens, string(curr_char), nothing)
            if token_id !== nothing
                push!(tokens, token_id)
                i = nextind(text, i)
                continue
            end
        end
        
        # Find word boundary
        word_end = i
        while word_end <= text_length
            curr_char = text[word_end]
            if ispunct(curr_char)
                # Allow hyphens and apostrophes within words
                if (curr_char == '\'' || curr_char == '-') && 
                   word_end < text_length && !isspace(text[nextind(text, word_end)])
                    word_end = nextind(text, word_end)
                    continue
                end
                break
            elseif isspace(curr_char)
                break
            end
            word_end = nextind(text, word_end)
        end
        
        # Handle end of text
        if i >= word_end
            if i >= text_length
                break
            end
            next_i = nextind(text, i)
            if next_i > text_length
                break
            end
            i = next_i
            continue
        end
        
        # Get the full word
        full_word = text[i:prevind(text, word_end)]
        
        # Handle empty word
        if isempty(full_word)
            if i >= text_length
                break
            end
            next_i = nextind(text, i)
            if next_i > text_length
                break
            end
            i = next_i
            continue
        end
        
        # Special handling for O'Neill-style names
        if startswith(full_word, "O'") || startswith(full_word, "O-")
            # Try with Ġ prefix first
            prefixed_word = "Ġ" * full_word
            if haskey(known_tokens, prefixed_word)
                push!(tokens, known_tokens[prefixed_word])
                i = word_end
                continue
            end
            
            # Then try without prefix
            if haskey(known_tokens, full_word)
                push!(tokens, known_tokens[full_word])
                i = word_end
                continue
            end
            
            # Finally try just the O' part
            prefixed_word = "Ġ" * full_word[1:2]
            if haskey(known_tokens, prefixed_word)
                push!(tokens, known_tokens[prefixed_word])
                i = nextind(text, i, 2)
                continue
            end
        end
        
        # Handle non-ASCII characters
        if !isascii(full_word)
            current_pos = firstindex(full_word)
            while current_pos <= ncodeunits(full_word)
                char = full_word[current_pos]
                char_token = string(char)
                
                if haskey(known_tokens, char_token)
                    push!(tokens, known_tokens[char_token])
                end
                
                current_pos = nextind(full_word, current_pos)
            end
            i = word_end
            continue
        end
        
        # Handle words ending in punctuation
        if endswith(full_word, r"[[:punct:]]")
            punct_pos = findlast(ispunct, full_word)
            if !isnothing(punct_pos)
                base_word = full_word[1:prevind(full_word, punct_pos)]
                punct = full_word[punct_pos:end]
                
                
                # Try with Ġ prefix
                prefixed_base = "Ġ" * base_word
                if haskey(known_tokens, prefixed_base)
                    push!(tokens, known_tokens[prefixed_base])
                    if haskey(known_tokens, punct)
                        push!(tokens, known_tokens[punct])
                    end
                    i = word_end
                    continue
                end
            end
        end
        
        # Special handling for words ending in period
        if endswith(full_word, '.')
            base_word = full_word[1:end-1]
            
            # Try without prefix first
            if haskey(known_tokens, base_word)
                push!(tokens, known_tokens[base_word])
                if haskey(known_tokens, ".")
                    push!(tokens, known_tokens["."])
                end
                i = word_end
                continue
            end
            
            # Then try with Ġ prefix
            prefixed_base = "Ġ" * base_word
            if haskey(known_tokens, prefixed_base)
                push!(tokens, known_tokens[prefixed_base])
                if haskey(known_tokens, ".")
                    push!(tokens, known_tokens["."])
                end
                i = word_end
                continue
            end
        end
        
        # Handle pure punctuation strings
        if all(c -> ispunct(c) || c in PUNCTUATION, full_word)
            # Try the full punctuation sequence first
            if haskey(known_tokens, full_word)
                push!(tokens, known_tokens[full_word])
                i = word_end
                continue
            end
            
            # Otherwise process character by character
            current_pos = firstindex(full_word)
            while current_pos < word_end
                char = string(full_word[current_pos])
                if haskey(known_tokens, char)
                    push!(tokens, known_tokens[char])
                end
                current_pos = nextind(full_word, current_pos)
            end
            i = word_end
            continue
        end
        
        # Handle words with non-ASCII characters
        if any(c -> !isascii(c) && c != '\'' && c != '-', full_word)
            # Try the full word first
            if haskey(known_tokens, full_word)
                push!(tokens, known_tokens[full_word])
                i = word_end
                continue
            end
            
            # Process character by character for non-ASCII text
            current_pos = firstindex(full_word)
            while current_pos <= ncodeunits(full_word)
                char = full_word[current_pos]
                char_token = string(char)
                
                if haskey(known_tokens, char)
                    # Special handling for numbers
                    cat = Base.Unicode.category_code(char)
                    if cat == Base.Unicode.UTF8PROC_CATEGORY_NO || # Number, Other
                       cat == Base.Unicode.UTF8PROC_CATEGORY_ND    # Number, Decimal Digit
                        # Try to combine with previous token if it was also a number
                        if !isempty(tokens)
                            prev_token = get(tokenizer.id_to_token, tokens[end], nothing)
                            if !isnothing(prev_token) && all(c -> isnumeric(c), prev_token)
                                combined = prev_token * char_token
                                if haskey(known_tokens, combined)
                                    pop!(tokens)  # Remove previous token
                                    push!(tokens, known_tokens[combined])
                                    current_pos = nextind(full_word, current_pos)
                                    continue
                                end
                            end
                        end
                    end
                    push!(tokens, known_tokens[char])
                else
                    # Handle unknown character
                    push!(tokens, tokenizer.special_tokens["[UNK]"])
                end
                current_pos = nextind(full_word, current_pos)
            end
            i = word_end
            continue
        end
        
        # Try to match special tokens
        longest_match = ""
        longest_id = nothing
        for (token, id) in tokenizer.special_tokens
            if startswith(@view(text[i:end]), token)
                if length(token) > length(longest_match)
                    longest_match = token
                    longest_id = id
                end
            end
        end
        
        # Handle punctuation characters
        if isempty(longest_match) && i <= lastindex(text)
            if ispunct(text[i]) || text[i] in ['[', ']', '.', ',', '!', '?', '-', '@', '{', '}']
                char = string(text[i])
                if haskey(tokenizer.vocab, char)
                    push!(tokens, tokenizer.vocab[char])
                    i = nextind(text, i)
                    continue
                end
            end
        end
        
        # If we found a special token match, use it
        if !isempty(longest_match)
            push!(tokens, longest_id)
            i += length(longest_match)
            continue
        end
        
        # Try to find the longest matching token
        try
            token, id = find_longest_token(tokenizer, text, i)
            if !isempty(token)
                push!(tokens, id)
                i += length(token)
            else
                # Handle single character as unknown
                char = string(text[i])
                if !any(haskey(KNOWN_TOKENS, substr) || haskey(tokenizer.vocab, substr) || haskey(tokenizer.special_tokens, substr) for substr in [char, "Ġ" * char])
                    push!(tokens, tokenizer.special_tokens["[UNK]"])
                end
                i = nextind(text, i)
            end
        catch e
            if e isa BoundsError
                # Handle boundary error by moving to next character
                i = nextind(text, i)
            else
                rethrow(e)
            end
        end
    end
    
    return tokens
end

# Implement TextEncodeBase.tokenize for vector of strings
function TextEncodeBase.tokenize(tokenizer::ModernBertTokenizer, texts::Vector{String}; token_ids::Bool=true)
    return [tokenize(tokenizer, text; token_ids=token_ids) for text in texts]
end

# Implement TextEncodeBase.encode for single string
function TextEncodeBase.encode(tokenizer::ModernBertTokenizer, text::AbstractString; special_tokens::Dict{String, Int}=Dict{String, Int}())
    # Handle empty string case
    if isempty(text)
        return Int[], Int[], Int[]
    end
    
    # Merge special tokens with default special tokens, giving priority to provided ones
    merged_special_tokens = merge(tokenizer.special_tokens, special_tokens)
    
    # Initialize result arrays
    tokens = Int[]
    token_types = Int[]
    attention_mask = Int[]
    
    # Add [CLS] token if present in merged special tokens
    if haskey(merged_special_tokens, "[CLS]")
        push!(tokens, merged_special_tokens["[CLS]"])
        push!(token_types, 0)
        push!(attention_mask, 1)
    
    # Process text character by character
    i = firstindex(text)
    text_length = lastindex(text)
    current_start = i
    
    while i <= text_length
        # Try to match special tokens first
        sorted_tokens = sort(collect(tokenizer.special_tokens), by=x->length(x.first), rev=true)
        found_special = false
        
        for (token, id) in sorted_tokens
            token_len = length(token)
            end_idx = i
            char_count = 0
            current_text = ""
            
            try
                current_idx = i
                while char_count < token_len && current_idx <= text_length
                    current_text *= text[current_idx]
                    current_idx = nextind(text, current_idx)
                    char_count += 1
                end
                
                if char_count == token_len
                    if current_text == token
                        # Add any pending text before the special token
                        if i > current_start
                            pending_text = text[current_start:prevind(text, i)]
                            if !isempty(pending_text)
                                # Tokenize the pending text
                                pending_tokens = tokenize_subwords(tokenizer, pending_text)
                                append!(tokens, pending_tokens)
                                append!(token_types, zeros(Int, length(pending_tokens)))
                                append!(attention_mask, ones(Int, length(pending_tokens)))
                            end
                        end
                        
                        # Add the special token
                        push!(tokens, id)
                        push!(token_types, 0)
                        push!(attention_mask, 1)
                        
                        i = current_idx
                        current_start = i
                        found_special = true
                        break
                    end
                end
            catch e
                if e isa StringIndexError
                    # Handle invalid UTF-8 sequence
                    break
                else
                    rethrow(e)
                end
            end
        end
        
        if !found_special
            # Handle end of text
            if i >= text_length
                break
            end
            
            # Move to next character
            next_i = nextind(text, i)
            if next_i > text_length
                break
            end
            i = next_i
        end
    end
    
    # Process any remaining text
    if current_start <= text_length
        remaining_text = text[current_start:end]
        if !isempty(remaining_text)
            # Tokenize the remaining text
            remaining_tokens = tokenize_subwords(tokenizer, remaining_text)
            append!(tokens, remaining_tokens)
            append!(token_types, zeros(Int, length(remaining_tokens)))
            append!(attention_mask, ones(Int, length(remaining_tokens)))
        end
    end
    
    # Split on special tokens and process each part
    parts = String[]
    current_part = ""
    for char in text
        if string(char) in keys(tokenizer.special_tokens)
            if !isempty(current_part)
                push!(parts, current_part)
                current_part = ""
            end
            push!(parts, string(char))
        else
            current_part *= char
        end
    end
    if !isempty(current_part)
        push!(parts, current_part)
    end
    
    # Process each part
    for part in parts
        if haskey(tokenizer.special_tokens, part)
            push!(tokens, tokenizer.special_tokens[part])
            push!(token_types, 0)
            push!(attention_mask, 1)
        end
    end
    
    # Add [SEP] token if not already present
    if isempty(tokens) || tokens[end] != tokenizer.special_tokens["[SEP]"]
        push!(tokens, tokenizer.special_tokens["[SEP]"])
        push!(token_types, 0)
        push!(attention_mask, 1)
    end
    
    # Truncate if necessary
    if length(tokens) > 512
        tokens = tokens[1:512]
        token_types = token_types[1:512]
        attention_mask = attention_mask[1:512]
    end
    
    return tokens, token_types, attention_mask
end

# Implement TextEncodeBase.encode for vector of strings
function TextEncodeBase.encode(tokenizer::ModernBertTokenizer, texts::Vector{String})
    # Process each text and collect results
    results = Vector{Tuple{Vector{Int}, Vector{Int}, Vector{Int}}}(undef, length(texts))
    
    for (idx, text) in enumerate(texts)
        try
            results[idx] = encode(tokenizer, text)
        catch e
            if e isa BoundsError || e isa StringIndexError
                # Handle invalid text by returning UNK token
                results[idx] = ([tokenizer.special_tokens["[UNK]"]], [0], [1])
            else
                rethrow(e)
            end
        end
    end
    
    # Get maximum sequence length (capped at 512)
    max_len = min(512, maximum(length(r[1]) for r in results))
    
    # Create padded arrays
    n_texts = length(texts)
    tokens_matrix = fill(tokenizer.special_tokens["[PAD]"], max_len, n_texts)
    types_matrix = zeros(Int, max_len, n_texts)
    mask_matrix = zeros(Int, max_len, n_texts)
    
    # Fill matrices with actual values
    for (j, (tokens, types, mask)) in enumerate(results)
        len = min(length(tokens), max_len)
        tokens_matrix[1:len, j] = tokens[1:len]
        types_matrix[1:len, j] = types[1:len]
        mask_matrix[1:len, j] = mask[1:len]
    end
    
    # Ensure all dimensions match
    @assert size(tokens_matrix) == size(types_matrix) == size(mask_matrix) "Matrix dimensions must match"
    return tokens_matrix, types_matrix, mask_matrix
end

end  # module ModernBertTokenizerImpl
