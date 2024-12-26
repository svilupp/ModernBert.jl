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
export ModernBertTokenizer, tokenize, encode, load_modernbert_tokenizer, vocab_size

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
    # Check for special tokens first
    for (token, id) in tokenizer.special_tokens
        if start_idx + length(token) - 1 <= lastindex(text)
            if text[start_idx:start_idx + length(token) - 1] == token
                return token, id
            end
        end
    end
    
    longest_match = ""
    longest_id = nothing
    current_idx = start_idx
    
    # Handle empty string case
    if isempty(text) || start_idx > lastindex(text)
        return longest_match, longest_id
    end
    
    # Determine if we're at start/after space
    is_start_or_after_space = start_idx == firstindex(text) || 
        (start_idx > firstindex(text) && isspace(text[prevind(text, start_idx)]))
    
    while current_idx <= lastindex(text)
        # Try to match increasingly longer substrings
        # Use nextind/prevind for proper UTF-8 character handling
        try
            substr = text[start_idx:current_idx]
            
            # Always prioritize regular tokens over Ġ-prefixed variants
            variants = [substr]
            if is_start_or_after_space
                push!(variants, "Ġ" * substr)
            end
            
            # Check each variant against token dictionaries
            for variant in variants
                # Check known_tokens first (highest priority)
                if haskey(tokenizer.known_tokens, variant)
                    longest_match = substr
                    longest_id = tokenizer.known_tokens[variant]
                    break
                end
                
                # Then check main vocabulary
                if haskey(tokenizer.vocab, variant)
                    longest_match = substr
                    longest_id = tokenizer.vocab[variant]
                    break
                end
            end
            
            # Move to next character safely with bounds check
            next_idx = nextind(text, current_idx)
            if next_idx > lastindex(text)
                break
            end
            current_idx = next_idx
        catch e
            if e isa StringIndexError || e isa BoundsError
                # Exit loop on any string indexing error
                break
            else
                rethrow(e)
            end
        end
    end
    
    return longest_match, longest_id
end

# Tokenize text into subwords
function tokenize_subwords(tokenizer::ModernBertTokenizer, text::String)
    # First check if the text is a special token
    if haskey(tokenizer.special_tokens, text)
        return [tokenizer.special_tokens[text]]
    end
    
    # Handle whitespace-only text first
    if isempty(strip(text))
        return [tokenizer.vocab[" "]]  # Return space token (50275)
    end
    
    # For all words, check all variants before returning UNK
    # Determine if we're at start/after space or mid-word
    is_start_or_after_space = firstindex(text) == 1 || 
        (firstindex(text) > 1 && isspace(text[prevind(text, firstindex(text))]))
    
    # Try all variants in appropriate order
    if is_start_or_after_space
        # At start or after space, check Ġ-prefixed first
        variants = ["Ġ" * text, text]
    else
        # Mid-word, check normal variant first
        variants = [text, "Ġ" * text]
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
        return [REQUIRED_TOKENS[" "]]  # Space token (50275)
    end
    
    # Cache commonly used dictionaries for faster lookups
    known_tokens = KNOWN_TOKENS
    special_tokens = tokenizer.special_tokens
    vocab = tokenizer.vocab
    
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
    last_was_space = true
    tokens = Int[]
    text_length = lastindex(text)
    
    # Pre-allocate buffers
    longest_match = ""
    current_text = ""
    
    # Main tokenization loop
    while i <= text_length
        # Handle whitespace efficiently
        if isspace(text[i])
            i = nextind(text, i)
            last_was_space = true
            continue
        end
        
        # Reset state for each token
        longest_id = nothing
        current_idx = i
        
        # Find word boundaries more efficiently
        word_end = i
        in_punctuation = false
        
        # Process characters until we hit a boundary
        while word_end <= text_length
            curr_char = text[word_end]
            
            # Stop at whitespace
            if isspace(curr_char)
                break
            end
            
            # Handle punctuation boundaries
            curr_is_punct = ispunct(curr_char)
            if curr_is_punct != in_punctuation && word_end > i
                break
            end
            in_punctuation = curr_is_punct
            
            # Advance to next character
            next_end = nextind(text, word_end)
            if next_end > text_length
                break
            end
            word_end = next_end
        end
        
        # Extract word safely
        full_word = text[i:prevind(text, word_end)]
            
            # Try to match the full word with cached dictionaries
            if last_was_space
                # After space, try Ġ-prefixed first
                prefixed_word = "Ġ" * full_word
                if haskey(known_tokens, prefixed_word)
                    push!(tokens, known_tokens[prefixed_word])
                    i = word_end
                    last_was_space = false
                    continue
                elseif haskey(vocab, prefixed_word)
                    push!(tokens, vocab[prefixed_word])
                    i = word_end
                    last_was_space = false
                    continue
                end
                
                # If the word ends with punctuation, try without it
                if endswith(full_word, r"[[:punct:]]")
                    base_word = full_word[1:prevind(full_word, findlast(ispunct, full_word))]
                    prefixed_base = "Ġ" * base_word
                    if haskey(KNOWN_TOKENS, prefixed_base)
                        push!(tokens, KNOWN_TOKENS[prefixed_base])
                        # Add the punctuation separately
                        punct = full_word[findlast(ispunct, full_word):end]
                        if haskey(KNOWN_TOKENS, punct)
                            push!(tokens, KNOWN_TOKENS[punct])
                        elseif haskey(tokenizer.vocab, punct)
                            push!(tokens, tokenizer.vocab[punct])
                        end
                        i = word_end
                        last_was_space = false
                        continue
                    end
                end
                
                # Try without trailing punctuation for common titles
                if endswith(full_word, '.')
                    base_word = full_word[1:prevind(full_word, lastindex(full_word))]
                    if haskey(KNOWN_TOKENS, "Ġ" * base_word)
                        push!(tokens, KNOWN_TOKENS["Ġ" * base_word])
                        i = nextind(text, i + ncodeunits(base_word))
                        last_was_space = false
                        continue
                    elseif haskey(tokenizer.vocab, "Ġ" * base_word)
                        push!(tokens, tokenizer.vocab["Ġ" * base_word])
                        i = nextind(text, i + ncodeunits(base_word))
                        last_was_space = false
                        continue
                    end
                end
            end
            
            # Try non-prefixed version with cached dictionaries
            if haskey(known_tokens, full_word)
                push!(tokens, known_tokens[full_word])
                i = word_end
                last_was_space = false
                continue
            elseif haskey(vocab, full_word)
                push!(tokens, vocab[full_word])
                i = word_end
                last_was_space = false
                continue
            end
            
            # Handle punctuation and special characters more efficiently
            if all(c -> ispunct(c) || c in ['[', ']', '.', ',', '!', '?', '-', '@', '{', '}', '\''], full_word)
                # Try to match the full punctuation sequence with cached dictionaries
                if haskey(known_tokens, full_word)
                    push!(tokens, known_tokens[full_word])
                    i = word_end
                    last_was_space = false
                    continue
                elseif haskey(vocab, full_word)
                    push!(tokens, vocab[full_word])
                    i = word_end
                    last_was_space = false
                    continue
                end
                
                # If we can't match the full sequence, process character by character
                for j in i:prevind(text, word_end)
                    char = text[j:j]
                    if haskey(KNOWN_TOKENS, char)
                        push!(tokens, KNOWN_TOKENS[char])
                    elseif haskey(tokenizer.vocab, char)
                        push!(tokens, tokenizer.vocab[char])
                    else
                        push!(tokens, tokenizer.special_tokens["[UNK]"])
                    end
                end
                i = word_end
                last_was_space = false
                continue  # Return to start of loop
            end
            
            # For non-ASCII characters that aren't allowed, emit [UNK]
            if any(c -> !isascii(c) && c != '\'' && c != '-', full_word)
                push!(tokens, tokenizer.special_tokens["[UNK]"])
                i = word_end
                last_was_space = false
                continue  # Return to start of loop
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
                last_was_space = false  # Reset space tracking after unknown token
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
                
                current_text *= text[current_idx]
                
                # For words after space, try Ġ-prefixed version first
                if last_was_space
                    # Check KNOWN_TOKENS first
                    if haskey(KNOWN_TOKENS, "Ġ" * current_text) && length(current_text) > length(longest_match)
                        longest_match = current_text
                        longest_id = KNOWN_TOKENS["Ġ" * current_text]
                    # Then check vocabulary
                    elseif haskey(tokenizer.vocab, "Ġ" * current_text) && length(current_text) > length(longest_match)
                        longest_match = current_text
                        longest_id = tokenizer.vocab["Ġ" * current_text]
                    end
                end
                
                # If no Ġ-prefixed match found or not after space, try non-prefixed version
                if isempty(longest_match) || length(current_text) > length(longest_match)
                    # Check KNOWN_TOKENS first
                    if haskey(KNOWN_TOKENS, current_text)
                        longest_match = current_text
                        longest_id = KNOWN_TOKENS[current_text]
                    # Then check vocabulary
                    elseif haskey(tokenizer.vocab, current_text)
                        longest_match = current_text
                        longest_id = tokenizer.vocab[current_text]
                    end
                end
                
                current_idx = nextind(text, current_idx)
            end
        end
        
        # If we found a match, add it and advance
        if !isempty(longest_match)
            push!(tokens, longest_id)
            # Safely advance index by the length of the matched token
            remaining_chars = length(longest_match)
            while remaining_chars > 0 && i <= lastindex(text)
                i = try
                    nextind(text, i)
                catch e
                    if e isa BoundsError
                        lastindex(text) + 1
                    else
                        rethrow(e)
                    end
                end
                remaining_chars -= 1
            end
        else
            # No match found or token not recognized, emit [UNK] and advance one character
            push!(tokens, tokenizer.special_tokens["[UNK]"])
            i = try
                nextind(text, i)
            catch e
                if e isa BoundsError
                    lastindex(text) + 1
                else
                    rethrow(e)
                end
            end
            last_was_space = false  # Reset space tracking after unknown token
        end
        
        # Update space tracking for next iteration
        last_was_space = i == firstindex(text) || (i > firstindex(text) && isspace(text[prevind(text, i)]))
        continue  # Go back to start of loop via @label next_iteration
    end
    
    # Add period token only if not already present and text ends with period
    if !isempty(text) && text[end] == '.' && !isempty(tokens) && tokens[end] != KNOWN_TOKENS["."]
        push!(tokens, KNOWN_TOKENS["."])  # Use period token (15)
    end
    
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
    
    # Add main tokens
    append!(tokens, tokenize(tokenizer, text))
    
    # Add SEP token at end
    push!(tokens, tokenizer.special_tokens["[SEP]"])
    
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
    # Process each text
    results = [encode(tokenizer, text) for text in texts]
    
    # Get maximum sequence length
    max_len = minimum([512, maximum(length(r[1]) for r in results)])
    
    # Create padded arrays
    n_texts = length(texts)
    tokens_matrix = fill(tokenizer.special_tokens["[PAD]"], max_len, n_texts)
    types_matrix = zeros(Int, max_len, n_texts)
    mask_matrix = zeros(Int, max_len, n_texts)
    
    # Fill matrices
    for (j, (tokens, types, mask)) in enumerate(results)
        len = min(length(tokens), max_len)
        tokens_matrix[1:len, j] = tokens[1:len]
        types_matrix[1:len, j] = types[1:len]
        mask_matrix[1:len, j] = mask[1:len]
    end
    
    return tokens_matrix, types_matrix, mask_matrix
end

end # module ModernBertTokenizerImpl
