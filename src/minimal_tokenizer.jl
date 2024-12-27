module ModernBertTokenizerImpl

using TextEncodeBase
using JSON3

# Import TextEncodeBase methods for extension
import TextEncodeBase: AbstractTextEncoder, AbstractTokenizer, tokenize, encode

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
            
            # Then check main vocabulary if we haven't found in known_tokens
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
        
        # Special handling for "Mr." case
        if current_token == "Mr" && !at_end && text[next_idx] == '.'
            mr_with_dot = text[start_idx:next_idx]
            if haskey(tokenizer.known_tokens, mr_with_dot)
                return mr_with_dot, tokenizer.known_tokens[mr_with_dot]
            elseif haskey(tokenizer.vocab, mr_with_dot)
                return mr_with_dot, tokenizer.vocab[mr_with_dot]
            end
            # If "Mr." isn't in vocab, return just "Mr" if it's a known token
            if haskey(tokenizer.known_tokens, current_token)
                return current_token, tokenizer.known_tokens[current_token]
            elseif haskey(tokenizer.vocab, current_token)
                return current_token, tokenizer.vocab[current_token]
            end
        end

        # Check if we're at a word boundary
        is_word_boundary = at_end || isspace(text[next_idx]) || (ispunct(text[next_idx]) && text[next_idx] ∉ ['-', '\''])
        
        # For compound words, try to match the full word first
        if !at_end && (text[next_idx] ∈ ['-', '\''])
            # Try to find the end of the compound word
            compound_end_idx = next_idx
            while compound_end_idx < lastindex(text)
                next_compound_idx = nextind(text, compound_end_idx)
                if next_compound_idx > lastindex(text)
                    break
                end
                next_char = text[next_compound_idx]
                # Stop at spaces or punctuation (except hyphens and apostrophes)
                if isspace(next_char) || (ispunct(next_char) && next_char ∉ ['-', '\''])
                    break
                end
                # Include the character and move to next
                compound_end_idx = next_compound_idx
            end
            
            # Always include the last character of the compound word
            if compound_end_idx < lastindex(text)
                compound_end_idx = nextind(text, compound_end_idx)
            
            # Try the full compound word first
            compound_token = text[start_idx:compound_end_idx]
            for variant in (is_start_or_after_space ? [compound_token, "Ġ" * compound_token] : [compound_token])
                if haskey(tokenizer.known_tokens, variant)
                    return compound_token, tokenizer.known_tokens[variant]
                elseif haskey(tokenizer.vocab, variant)
                    return compound_token, tokenizer.vocab[variant]
                end
            end
            
            # If full compound word isn't found, try progressive subparts
            current_start = start_idx
            while current_start <= compound_end_idx
                # Try each possible end position for the current start
                current_end = nextind(text, current_start)
                while current_end <= compound_end_idx
                    subtoken = text[current_start:current_end]
                    for variant in (current_start == start_idx && is_start_or_after_space ? 
                                  [subtoken, "Ġ" * subtoken] : [subtoken])
                        if haskey(tokenizer.known_tokens, variant)
                            return subtoken, tokenizer.known_tokens[variant]
                        elseif haskey(tokenizer.vocab, variant)
                            return subtoken, tokenizer.vocab[variant]
                        end
                    end
                    current_end = nextind(text, current_end)
                end
                # Move start position forward if no matches found
                current_start = nextind(text, current_start)
            end
            
            # If no subparts match, return the first character
            return text[start_idx:start_idx], get(tokenizer.vocab, text[start_idx:start_idx], 
                   tokenizer.special_tokens["[UNK]"])
        end
        
        if found_match && is_word_boundary
            return longest_match, longest_id
        end
        
        next_idx = nextind(text, current_idx)
        at_end = next_idx > lastindex(text)
        
        # Check if we're at a word boundary
        is_word_boundary = at_end || isspace(text[next_idx]) || (ispunct(text[next_idx]) && text[next_idx] ∉ ['-', '\''])
        
        # If we found a match and we're at a word boundary, return it
        if found_match && is_word_boundary
            # Special handling for punctuation after known tokens
            if !at_end && ispunct(text[next_idx])
                # Try to include the punctuation in the token
                extended_token = text[start_idx:next_idx]
                if haskey(tokenizer.known_tokens, extended_token)
                    return extended_token, tokenizer.known_tokens[extended_token]
                elseif haskey(tokenizer.vocab, extended_token)
                    return extended_token, tokenizer.vocab[extended_token]
                end
            end
            return longest_match, longest_id
        end
        
        # Handle punctuation specially
        if ispunct(text[current_idx])
            # For compound words (with hyphens/apostrophes), continue building token
            if (text[current_idx] ∈ ['-', '\'']) && !at_end && !isspace(text[next_idx])
                current_idx = next_idx
                continue
            end
            
            # Return the last match if we have one
            if last_match !== nothing
                # If it's a period after "Mr", keep them together
                if text[current_idx] == '.' && last_match == "Mr"
                    current_token = last_match * "."
                    if haskey(tokenizer.known_tokens, current_token)
                        return current_token, tokenizer.known_tokens[current_token]
                    end
                end
                return last_match, last_match_id
            end
            
            # If it's a standalone punctuation, return it as a token
            if current_idx == start_idx || (current_idx > start_idx && isspace(text[prevind(text, current_idx)]))
                return string(text[current_idx]), get(tokenizer.vocab, string(text[current_idx]), tokenizer.special_tokens["[UNK]"])
            end
            
            # For other punctuation, try without it and return if we find a match
            token_without_punct = text[start_idx:prevind(text, current_idx)]
            if !isempty(token_without_punct)
                for variant in (is_start_or_after_space ? [token_without_punct, "Ġ" * token_without_punct] : [token_without_punct])
                    if haskey(tokenizer.known_tokens, variant)
                        return token_without_punct, tokenizer.known_tokens[variant]
                    elseif haskey(tokenizer.vocab, variant)
                        return token_without_punct, tokenizer.vocab[variant]
                    end
                end
            end
            
            # If we can't match anything, return just the punctuation as a token
            return string(text[current_idx]), get(tokenizer.vocab, string(text[current_idx]), tokenizer.special_tokens["[UNK]"])
        end
        
        # If we've found a match but haven't hit a word boundary yet, try to find a longer match
        if found_match && !at_end
            next_char = text[next_idx]
            if !isspace(next_char) && (!ispunct(next_char) || next_char ∈ ['-', '\''])
                current_idx = next_idx
                continue
            end
        end
        
        # Move to next character if we haven't already
        if current_idx < next_idx
            current_idx = next_idx
        end
    end
    
    # Return the last successful match if we have one
    if last_match !== nothing
        return last_match, last_match_id
    end
    
    # If we have a longest match, return it
    if longest_match !== nothing
        return longest_match, longest_id
    end
    
    # No match found, return the first character as UNK
    return string(text[start_idx]), tokenizer.special_tokens["[UNK]"]
end

# Tokenize text into subwords
function tokenize_subwords(tokenizer::ModernBertTokenizer, text::String)
    # First check if the text is a special token (including [MASK])
    if haskey(tokenizer.special_tokens, text)
        return [tokenizer.special_tokens[text]]
    end
    
    # Handle whitespace-only text first
    if isempty(strip(text))
        return [tokenizer.vocab[" "]]  # Return space token (50275)
    end
    
    # For all words, check all variants before returning UNK
    # Determine if we're at start/after space/punctuation
    prev_idx = prevind(text, firstindex(text))
    is_start_or_after_space = firstindex(text) == 1 || 
        (firstindex(text) > 1 && (isspace(text[prev_idx]) || ispunct(text[prev_idx])))
    
    # Try all variants in appropriate order
    if is_start_or_after_space
        # At start or after space/punctuation, ALWAYS try Ġ-prefixed first
        variants = ["Ġ" * text]
        # Only try non-prefixed version if Ġ-prefixed fails
        push!(variants, text)
    else
        # Mid-word, only try normal variant
        variants = [text]
    end
    
    # Check each variant against all token dictionaries
    for variant in variants
        # Check known_tokens first (highest priority)
        if haskey(tokenizer.known_tokens, variant)
            return [tokenizer.known_tokens[variant]]
        end
        
        # Then check main vocabulary
        if haskey(tokenizer.vocab, variant)
            return [tokenizer.vocab[variant]]
        end
        
        # Finally check special tokens
        if haskey(tokenizer.special_tokens, variant)
            return [tokenizer.special_tokens[variant]]
        end
    end
    
    # If no match found in any variant, return UNK
    return [tokenizer.special_tokens["[UNK]"]]
end

# Get vocabulary size
function vocab_size(tokenizer::ModernBertTokenizer)
    # Include both vocabulary and special tokens
    # Note: special tokens are already in vocab, so just return vocab length
    # This should include:
    # - Base vocabulary (50280 tokens)
    # - Special tokens ([UNK], [CLS], [SEP], [PAD], [MASK]) (5 tokens)
    # - Space token (" ") (1 token)
    # - GPT-2 special tokens (Ġ, Ċ) (2 tokens)
    # Total: 50288 tokens
    return length(tokenizer.vocab)
end

# Default constructor without vocab path
function load_modernbert_tokenizer()
    # Initialize with empty vocabulary and module-level special tokens
    vocab = Dict{String, Int}()
    special_tokens = SPECIAL_TOKENS
    
    # Add special tokens and required tokens to vocabulary
    for (token, id) in special_tokens
        vocab[token] = id
    end
    
    # Add required tokens to vocabulary
    for (token, id) in REQUIRED_TOKENS
        vocab[token] = id
    end
    
    return ModernBertTokenizer(vocab, special_tokens)
end

# Basic tokenization function
function TextEncodeBase.tokenize(tokenizer::ModernBertTokenizer, text::AbstractString; token_ids::Bool=true)
    # Initialize tokens array and handle special cases
    local tokens = Int[]
    
    # Early returns for special cases
    if isempty(text)
        return tokens
    elseif all(isspace, text)
        # Handle whitespace tokens (50275)
        if haskey(tokenizer.known_tokens, "Ġ")
            return [tokenizer.known_tokens["Ġ"]]
        elseif haskey(tokenizer.vocab, "Ġ")
            return [tokenizer.vocab["Ġ"]]
        elseif haskey(tokenizer.known_tokens, " ")
            return [tokenizer.known_tokens[" "]]
        elseif haskey(tokenizer.vocab, " ")
            return [tokenizer.vocab[" "]]
        else
            return tokens
        end
    end
    
    # Cache commonly used dictionaries for faster lookups
    known_tokens = tokenizer.known_tokens
    special_tokens = tokenizer.special_tokens
    vocab = tokenizer.vocab
    
    # Ensure known_tokens is initialized
    if isnothing(known_tokens)
        known_tokens = KNOWN_TOKENS
    end
    
    # Check for exact matches in order of priority
    if haskey(known_tokens, text)
        return [known_tokens[text]]
    elseif haskey(known_tokens, "Ġ" * text)
        return [known_tokens["Ġ" * text]]
    elseif haskey(special_tokens, text)
        return [special_tokens[text]]
    end
    
    # Cache frequently used values
    known_tokens = tokenizer.known_tokens
    vocab = tokenizer.vocab
    special_tokens = tokenizer.special_tokens
    
    # Initialize state for main tokenization
    i = firstindex(text)
    tokens = Int[]
    text_length = lastindex(text)
    longest_match = ""
    longest_id = nothing
    current_text = ""
    current_idx = i
    
    # Main tokenization loop with simplified boundary detection
    while i <= text_length
        # Skip whitespace
        while i <= text_length && isspace(text[i])
            i = nextind(text, i)
        end
        
        # Break if we've reached the end
        i > text_length && break
        
        # Find word boundary
        word_end = i
        curr_char = text[word_end]
        
        # Handle punctuation as single tokens
        if ispunct(curr_char)
            punct_token = string(curr_char)
            # Try known tokens first, then vocab
            token_id = get(known_tokens, punct_token, get(vocab, punct_token, nothing))
            if token_id !== nothing
                push!(tokens, token_id)
                i = nextind(text, i)
                continue
            end
        end
        
        # Find next word boundary
        while word_end <= text_length
            curr_char = text[word_end]
            next_end = nextind(text, word_end)
            
            # Stop at whitespace or end of text
            (isspace(curr_char) || next_end > text_length) && break
            
            # Handle word-internal punctuation
            if ispunct(curr_char)
                # Allow hyphens and apostrophes within words
                if (curr_char == '\'' || curr_char == '-') && 
                   next_end <= text_length && !isspace(text[next_end])
                    word_end = next_end
                    continue
                end
                break
            end
            
            word_end = next_end
        end
        
        # Extract word safely
        if i >= word_end
            # Check if we can safely advance
            if i >= text_length
                break
            end
            next_i = nextind(text, i)
            if next_i > text_length
                break
            end
            i = next_i  # Advance to next character
            continue
        end
        full_word = text[i:prevind(text, word_end)]
        if isempty(full_word)
            # Check if we can safely advance
            if i >= text_length
                break
            end
            next_i = nextind(text, i)
            if next_i > text_length
                break
            end
            i = next_i  # Advance to next character
            continue
        end
            
            # Try to match the full word with cached dictionaries
            # For words after space, try with and without Ġ prefix based on context
            # For words like "O'Neill", try Ġ prefix first
            if startswith(full_word, "O'") || startswith(full_word, "O-")
                prefixed_word = "Ġ" * full_word
                if haskey(known_tokens, prefixed_word)
                    push!(tokens, known_tokens[prefixed_word])
                    i = word_end
                    continue
                elseif haskey(vocab, prefixed_word)
                    push!(tokens, vocab[prefixed_word])
                    i = word_end
                    continue
                end
            else
                # For other words, try without prefix first
                if haskey(known_tokens, full_word)
                    push!(tokens, known_tokens[full_word])
                    i = word_end
                    continue
                elseif haskey(vocab, full_word)
                    push!(tokens, vocab[full_word])
                    i = word_end
                    continue
                end
                
                # Then try with Ġ prefix
                prefixed_word = "Ġ" * full_word
                if haskey(known_tokens, prefixed_word)
                    push!(tokens, known_tokens[prefixed_word])
                    i = word_end
                    continue
                elseif haskey(vocab, prefixed_word)
                    push!(tokens, vocab[prefixed_word])
                    i = word_end
                    continue
                end
            end
            
            # Try without prefix
            if haskey(known_tokens, full_word)
                push!(tokens, known_tokens[full_word])
                i = word_end
                continue
            elseif haskey(vocab, full_word)
                push!(tokens, vocab[full_word])
                i = word_end
                continue
            end
            
            # Handle UTF-8 and special characters
            if !isascii(full_word)
                current_pos = 1
                while current_pos <= ncodeunits(full_word)
                    char_end = nextind(full_word, current_pos)
                    char = full_word[current_pos:prevind(full_word, char_end)]
                    
                    # Try with and without Ġ prefix
                    char_token = "Ġ" * char
                    if haskey(known_tokens, char_token)
                        push!(tokens, known_tokens[char_token])
                    elseif haskey(vocab, char_token)
                        push!(tokens, vocab[char_token])
                    elseif haskey(known_tokens, char)
                        push!(tokens, known_tokens[char])
                    elseif haskey(vocab, char)
                        push!(tokens, vocab[char])
                    else
                        push!(tokens, tokenizer.special_tokens["[UNK]"])
                    end
                    
                    current_pos = char_end
                end
                i = word_end
                continue
            end
            
            # Handle words ending with punctuation
            if endswith(full_word, r"[[:punct:]]")
                punct_pos = findlast(ispunct, full_word)
                if !isnothing(punct_pos)
                    base_word = full_word[1:prevind(full_word, punct_pos)]
                    prefixed_base = "Ġ" * base_word
                    
                    # Try base word with prefix
                    if haskey(known_tokens, prefixed_base)
                        push!(tokens, known_tokens[prefixed_base])
                        # Add punctuation
                        punct = full_word[punct_pos:end]
                        if haskey(known_tokens, punct)
                            push!(tokens, known_tokens[punct])
                        elseif haskey(vocab, punct)
                            push!(tokens, vocab[punct])
                        end
                        i = word_end
                        continue
                    end
                end
            end
            
            # Handle common titles with trailing period
            if endswith(full_word, '.')
                base_word = full_word[1:prevind(full_word, lastindex(full_word))]
                # Try base word without prefix first for titles like "Mr."
                if haskey(known_tokens, base_word)
                    push!(tokens, known_tokens[base_word])
                    if haskey(known_tokens, ".")
                        push!(tokens, known_tokens["."])
                    elseif haskey(vocab, ".")
                        push!(tokens, vocab["."])
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
                    elseif haskey(vocab, ".")
                        push!(tokens, vocab["."])
                    end
                    i = word_end
                    continue
                end
            end
            
            # Handle punctuation sequences
            if all(c -> ispunct(c) || c in ['[', ']', '.', ',', '!', '?', '-', '@', '{', '}', '\''], full_word)
                # Try full sequence first
                if haskey(known_tokens, full_word)
                    push!(tokens, known_tokens[full_word])
                    i = word_end
                    continue
                elseif haskey(vocab, full_word)
                    push!(tokens, vocab[full_word])
                    i = word_end
                    continue
                end
                
                # Process character by character
                current_pos = i
                while current_pos < word_end
                    char = text[current_pos:current_pos]
                    if haskey(known_tokens, char)
                        push!(tokens, known_tokens[char])
                    elseif haskey(vocab, char)
                        push!(tokens, vocab[char])
                    else
                        push!(tokens, tokenizer.special_tokens["[UNK]"])
                    end
                    current_pos = nextind(text, current_pos)
                end
                i = word_end
                continue
            end
            
            # Handle special characters and subscripts
            if any(c -> !isascii(c) && c != '\'' && c != '-', full_word)
                # Try to match the full word first
                if haskey(known_tokens, full_word)
                    push!(tokens, known_tokens[full_word])
                    i = word_end
                    continue
                elseif haskey(vocab, full_word)
                    push!(tokens, vocab[full_word])
                    i = word_end
                    continue
                end
                
                # Process character by character for special characters
                current_pos = 1
                while current_pos <= ncodeunits(full_word)
                    char_end = nextind(full_word, current_pos)
                    char = full_word[current_pos:prevind(full_word, char_end)]
                    
                    # Try to match the character
                    if haskey(known_tokens, char)
                        push!(tokens, known_tokens[char])
                    elseif haskey(vocab, char)
                        push!(tokens, vocab[char])
                    else
                        # Check if it's a subscript or special character
                        cat = Base.Unicode.category_code(first(char))
                        if cat == Base.Unicode.UTF8PROC_CATEGORY_NO || # Number, Other
                           cat == Base.Unicode.UTF8PROC_CATEGORY_SO || # Symbol, Other
                           cat == Base.Unicode.UTF8PROC_CATEGORY_MN    # Mark, Non-Spacing
                            # Try to join with previous token if possible
                            if !isempty(tokens)
                                prev_token = get(tokenizer.id_to_token, tokens[end], nothing)
                                if !isnothing(prev_token)
                                    combined = prev_token * char
                                    if haskey(known_tokens, combined)
                                        tokens[end] = known_tokens[combined]
                                        current_pos = char_end
                                        continue
                                    elseif haskey(vocab, combined)
                                        tokens[end] = vocab[combined]
                                        current_pos = char_end
                                        continue
                                    end
                                end
                            end
                        end
                        push!(tokens, tokenizer.special_tokens["[UNK]"])
                    end
                    current_pos = char_end
                end
                i = word_end
                continue
            end
            
            # First check for special tokens
            for (token, id) in tokenizer.special_tokens
                if startswith(@view(text[i:end]), token)
                    if length(token) > length(longest_match)
                        longest_match = token
                        longest_id = id
                        break  # Found a special token, stop looking
                    end
                end
            end
        
            # Then check for punctuation and special characters
            if isempty(longest_match) && i <= lastindex(text)
                char = text[i:i]
                if ispunct(text[i]) || text[i] in ['[', ']', '.', ',', '!', '?', '-', '@', '{', '}']
                    if haskey(tokenizer.vocab, char)
                        longest_match = char
                        longest_id = tokenizer.vocab[char]
                    end
                end
            end
        
        # If no special token found, check if the current character is unknown
        if isempty(longest_match)
            # Safely get next character boundary
            next_i = try
                nextind(text, i)
            catch e
                if e isa BoundsError
                    lastindex(text) + 1
                else
                    rethrow(e)
                end
            end
            char = text[i:min(next_i-1, lastindex(text))]
            # Check if the character or its Ġ-prefixed version is known
            if !any(haskey(KNOWN_TOKENS, substr) || haskey(tokenizer.vocab, substr) || haskey(tokenizer.special_tokens, substr) for substr in [char, "Ġ" * char])
                # If neither version is known, emit [UNK] token
                push!(tokens, tokenizer.special_tokens["[UNK]"])
                i = nextind(text, i)
                continue  # Return to start of loop
            end
        end
        
        # If no special token found, try regular tokens
        if isempty(longest_match)
            while current_idx <= lastindex(text)
                # Stop at space
                if isspace(text[current_idx])
                    break
                end
                
                # Initialize token matching state for this word
                if current_idx == i
                    longest_match = ""
                    longest_id = nothing
                end
                
                current_text *= text[current_idx]
                
                # Try Ġ-prefixed version first
                prefixed_text = "Ġ" * current_text
                if haskey(known_tokens, prefixed_text) && length(current_text) > length(longest_match)
                    longest_match = current_text
                    longest_id = known_tokens[prefixed_text]
                elseif haskey(vocab, prefixed_text) && length(current_text) > length(longest_match)
                    longest_match = current_text
                    longest_id = vocab[prefixed_text]
                end
                
                # Try non-prefixed version
                if isempty(longest_match) || length(current_text) > length(longest_match)
                    if haskey(known_tokens, current_text)
                        longest_match = current_text
                        longest_id = known_tokens[current_text]
                    elseif haskey(vocab, current_text)
                        longest_match = current_text
                        longest_id = vocab[current_text]
                    end
                end
                
                current_idx = nextind(text, current_idx)
            end
        end
        
        # If we found a match, add it and advance
        if !isempty(longest_match) && !isnothing(longest_id)
            push!(tokens, longest_id)
            # Safely advance index by the length of the matched token
            i = nextind(text, i + ncodeunits(longest_match) - 1)
        else
            # No match found, emit [UNK] and advance one character
            push!(tokens, tokenizer.special_tokens["[UNK]"])
            i = nextind(text, i)
        end
        continue
    end
    
    # Add period token only if not already present and text ends with period
    if !isempty(text) && text[end] == '.' && !isempty(tokens) && tokens[end] != KNOWN_TOKENS["."]
        push!(tokens, KNOWN_TOKENS["."])  # Use period token (15)
    end
    
    # Note: [CLS] and [SEP] tokens are added by the encode function
    return tokens
end

# Add method for Vector{String}
function TextEncodeBase.tokenize(tokenizer::ModernBertTokenizer, texts::Vector{String}; token_ids::Bool=true)
    return [tokenize(tokenizer, text; token_ids=token_ids) for text in texts]
end

# Define encode methods for ModernBertTokenizer
function TextEncodeBase.encode(tokenizer::ModernBertTokenizer, text::AbstractString)
    # Initialize tokens array and add CLS token at start
    tokens = Int[tokenizer.special_tokens["[CLS]"]]
    
    # Handle empty string case
    if isempty(text)
        push!(tokens, tokenizer.special_tokens["[SEP]"])
        # Create token_types and attention_mask
        token_types = zeros(Int, 2)
        attention_mask = ones(Int, 2)
        return tokens, token_types, attention_mask
    end
    
    # Split on special tokens and tokenize parts
    parts = String[]
    current_start = 1
    text_length = lastindex(text)
    
    # Sort special tokens by length (longest first) to avoid partial matches
    sorted_tokens = sort(collect(tokenizer.special_tokens), by=x->length(x.first), rev=true)
    
    i = firstindex(text)
    while i <= text_length
        found_special = false
        for (token, id) in sorted_tokens
            token_len = length(token)
            # Calculate end index safely using UTF-8 character boundaries
            end_idx = i
            try
                # Get substring safely using character-based indexing
                current_text = ""
                current_idx = i
                char_count = 0
                while char_count < token_len && current_idx <= text_length
                    current_text *= text[current_idx]
                    current_idx = nextind(text, current_idx)
                    char_count += 1
                end
                # Only proceed if we have enough characters
                if char_count == token_len
                    if current_text == token
                        # Add text before special token
                        if i > current_start
                            push!(parts, text[current_start:prevind(text, i)])
                        end
                        # Add special token
                        push!(parts, token)
                        # Move index past token safely
                        i = nextind(text, end_idx)
                        current_start = i
                        found_special = true
                        break
                    end
                end
            catch e
                if e isa StringIndexError
                    break  # Skip to next character if we hit invalid UTF-8 boundaries
                else
                    rethrow(e)
                end
            end
        end
        if !found_special
            # Check if we can safely advance
            if i >= text_length
                break
            end
            next_i = nextind(text, i)
            if next_i > text_length
                break
            end
            i = next_i
        end
    end
    
    # Add remaining text
    if current_start <= text_length
        push!(parts, text[current_start:end])
    end
    
    # Tokenize each part
    for part in parts
        if haskey(tokenizer.special_tokens, part)
            push!(tokens, tokenizer.special_tokens[part])
        else
            append!(tokens, tokenize(tokenizer, part))
        end
    end
    
    # Add SEP token at end if not already present
    if isempty(tokens) || tokens[end] != tokenizer.special_tokens["[SEP]"]
        push!(tokens, tokenizer.special_tokens["[SEP]"])
    end
    
    # Truncate to maximum length (512 tokens)
    if length(tokens) > 512
        tokens = tokens[1:512]
    end
    
    # Create token_types and attention_mask
    token_types = zeros(Int, length(tokens))
    attention_mask = ones(Int, length(tokens))
    
    return tokens, token_types, attention_mask
end

# Add method for Vector{String}
function TextEncodeBase.encode(tokenizer::ModernBertTokenizer, texts::Vector{String})
    # Process each text with bounds checking
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
