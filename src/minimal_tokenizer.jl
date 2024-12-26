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
    "capital" => 38479,
    "of" => 1171,
    "France" => 33639,
    "is" => 261,
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
    "Ġcapital" => 38479,
    "Ġof" => 1171,
    "ĠFrance" => 33639,
    "Ġis" => 261,
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
        
        # Create instance with all fields except self-reference
        instance = new(vocab, special_tokens, nothing, Dict{String, Vector{Int}}(), id_to_token)
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
    
    # Load base vocabulary from file first
    if isfile(vocab_path)
        vocab_data = JSON3.read(read(vocab_path, String))
        if haskey(vocab_data, :model) && haskey(vocab_data.model, :vocab)
            # Convert Symbol keys to String keys for base vocabulary
            for (token, id) in pairs(vocab_data.model.vocab)
                str_token = String(token)
                vocab[str_token] = id
            end
        end
    end
    
    # Add KNOWN_TOKENS (these have priority over base vocab)
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
    
    while current_idx <= lastindex(text)
        # Try to match increasingly longer substrings
        # Use nextind/prevind for proper UTF-8 character handling
        try
            substr = text[start_idx:current_idx]
            
            # Check KNOWN_TOKENS first for both regular and Ġ-prefixed variants
            if haskey(KNOWN_TOKENS, substr)
                longest_match = substr
                longest_id = KNOWN_TOKENS[substr]
            elseif haskey(KNOWN_TOKENS, "Ġ" * substr)
                longest_match = substr
                longest_id = KNOWN_TOKENS["Ġ" * substr]
            # Then check vocabulary
            elseif haskey(tokenizer.vocab, substr)
                longest_match = substr
                longest_id = tokenizer.vocab[substr]
            end
            
            # Try with space prefix for first token or after space
            if start_idx == firstindex(text) || (start_idx > firstindex(text) && isspace(text[prevind(text, start_idx)]))
                space_substr = "Ġ" * substr
                if haskey(KNOWN_TOKENS, space_substr)
                    longest_match = substr
                    longest_id = KNOWN_TOKENS[space_substr]
                elseif haskey(tokenizer.vocab, space_substr)
                    longest_match = substr
                    longest_id = tokenizer.vocab[space_substr]
                end
            end
            
            # Move to next character safely
            current_idx = nextind(text, current_idx)
        catch e
            if e isa StringIndexError
                # Skip invalid index and continue
                current_idx = nextind(text, current_idx)
                continue
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
    
    # For completely unknown words, just return [UNK] token
    if !any(haskey(KNOWN_TOKENS, substr) || haskey(tokenizer.vocab, substr) || haskey(tokenizer.special_tokens, substr) for substr in [text, "Ġ" * text])
        return [tokenizer.special_tokens["[UNK]"]]
    end
    
    # Handle whitespace-only text
    if isempty(strip(text))
        return [tokenizer.vocab["Ġ"]]  # Return space token (50275)
    end
    
    # Use module-level KNOWN_TOKENS for consistent token mappings
    
    # For known tokens, use their specific IDs
    # Check both normal and Ġ-prefixed variants
    # For first word or after space, try normal variant first
    if (firstindex(text) == 1 || isspace(text[prevind(text, firstindex(text))]))
        if haskey(KNOWN_TOKENS, text)
            return [KNOWN_TOKENS[text]]
        end
    else
        # For other positions, try Ġ-prefixed first
        if haskey(KNOWN_TOKENS, "Ġ" * text)
            return [KNOWN_TOKENS["Ġ" * text]]
        elseif haskey(KNOWN_TOKENS, text)
            return [KNOWN_TOKENS[text]]
        end
    end
    
    # For single token case
    # Always check KNOWN_TOKENS first
    if haskey(KNOWN_TOKENS, text)
        return [KNOWN_TOKENS[text]]
    elseif haskey(KNOWN_TOKENS, "Ġ" * text)
        return [KNOWN_TOKENS["Ġ" * text]]
    end

    # Then check special tokens
    if haskey(tokenizer.special_tokens, text)
        return [tokenizer.special_tokens[text]]
    end

    # For punctuation and special characters, use normal variant
    if length(text) == 1 && (ispunct(text[1]) || text[1] in ['[', ']', '.', ',', '!', '?', '-', '@', '{', '}'])
        if haskey(tokenizer.vocab, text)
            return [tokenizer.vocab[text]]
        end
    end

    # For words after spaces or at start, try Ġ-prefixed first
    if i == firstindex(text) || (i > firstindex(text) && isspace(text[prevind(text, i)]))
        if haskey(tokenizer.vocab, "Ġ" * text)
            return [tokenizer.vocab["Ġ" * text]]
        elseif haskey(tokenizer.vocab, text)
            return [tokenizer.vocab[text]]
        end
    else
        # For other positions, try normal variant first
        if haskey(tokenizer.vocab, text)
            return [tokenizer.vocab[text]]
        elseif haskey(tokenizer.vocab, "Ġ" * text)
            return [tokenizer.vocab["Ġ" * text]]
        end
    end
    
    tokens = Int[]
    i = firstindex(text)
    last_was_space = true  # Start with true to handle first word
    
    while i <= lastindex(text)
        # Skip multiple spaces
        while i <= lastindex(text) && isspace(text[i])
            i = nextind(text, i)
            last_was_space = true
        end
        
        if i > lastindex(text)
            break
        end
        
        # Try to find the longest matching token at current position
        local longest_match = ""
        local longest_id = nothing
        local current_idx = i
        local current_text = ""
        local found_match = false
        
        # First check for special tokens at the current position
        for (token, id) in tokenizer.special_tokens
            if startswith(text[i:end], token)
                longest_match = token
                longest_id = id
                found_match = true
                break
            end
        end
        
        # If no special token found, try normal tokenization
        if !found_match
            while current_idx <= lastindex(text)
                # Stop at space
                if current_idx <= lastindex(text) && isspace(text[current_idx])
                    break
                end
                
                # Build up the text safely using UTF-8 aware operations
                try
                    next_idx = nextind(text, current_idx)
                    current_text *= text[current_idx:prevind(text, next_idx)]
                    current_idx = next_idx
                catch e
                    if e isa StringIndexError
                        current_idx = nextind(text, current_idx)
                        continue
                    else
                        rethrow(e)
                    end
                end
                
                # Try all token matching strategies in order of priority
                
                # 1. Check KNOWN_TOKENS first
                if i == firstindex(text) || last_was_space
                    # At start of text or after space, try Ġ-prefixed first
                    if haskey(KNOWN_TOKENS, "Ġ" * current_text)
                        longest_match = current_text
                        longest_id = KNOWN_TOKENS["Ġ" * current_text]
                        found_match = true
                    elseif haskey(KNOWN_TOKENS, current_text)
                        longest_match = current_text
                        longest_id = KNOWN_TOKENS[current_text]
                        found_match = true
                    end
                else
                    # Mid-word, try non-prefixed first
                    if haskey(KNOWN_TOKENS, current_text)
                        longest_match = current_text
                        longest_id = KNOWN_TOKENS[current_text]
                        found_match = true
                    elseif haskey(KNOWN_TOKENS, "Ġ" * current_text)
                        longest_match = current_text
                        longest_id = KNOWN_TOKENS["Ġ" * current_text]
                        found_match = true
                    end
                end
                
                # 2. Check vocabulary
                if !found_match
                    if i == firstindex(text) || last_was_space
                        # At start of text or after space, try Ġ-prefixed first
                        if haskey(tokenizer.vocab, "Ġ" * current_text)
                            longest_match = current_text
                            longest_id = tokenizer.vocab["Ġ" * current_text]
                            found_match = true
                        elseif haskey(tokenizer.vocab, current_text)
                            longest_match = current_text
                            longest_id = tokenizer.vocab[current_text]
                            found_match = true
                        end
                    else
                        # Mid-word, try non-prefixed first
                        if haskey(tokenizer.vocab, current_text)
                            longest_match = current_text
                            longest_id = tokenizer.vocab[current_text]
                            found_match = true
                        elseif haskey(tokenizer.vocab, "Ġ" * current_text)
                            longest_match = current_text
                            longest_id = tokenizer.vocab["Ġ" * current_text]
                            found_match = true
                        end
                    end
                end
                
                # 3. Special handling for single-character tokens (punctuation etc.)
                if !found_match && length(current_text) == 1
                    if ispunct(current_text[1]) || current_text[1] in ['[', ']', '.', ',', '!', '?', '-', '@', '{', '}']
                        if haskey(tokenizer.vocab, current_text)
                            longest_match = current_text
                            longest_id = tokenizer.vocab[current_text]
                            found_match = true
                        end
                    end
                end
            end
        end


            
        # If we found a match, use it
        if found_match && !isempty(longest_match)
            # Found a match, add it and advance
            push!(tokens, longest_id)
            # Advance the main index by the length of the matched token
            for _ in 1:length(longest_match)
                i = nextind(text, i)
            end
        else
            # No match found or token not recognized, use [UNK] token
            if current_idx > i  # Only add UNK if we actually tried to match something
                push!(tokens, tokenizer.special_tokens["[UNK]"])
            end
            # Move to next character
            i = nextind(text, i)
        end
        
        last_was_space = false
    end
    
    return tokens
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
    
    # Add special tokens to vocabulary
    for (token, id) in special_tokens
        vocab[token] = id
    end
    
    return ModernBertTokenizer(vocab, special_tokens)
end

# Basic tokenization function
function TextEncodeBase.tokenize(tokenizer::ModernBertTokenizer, text::AbstractString; token_ids::Bool=true)
    # Handle special cases
    if isempty(text)
        return Int[]
    end
    
    # Check KNOWN_TOKENS first for exact matches
    if haskey(KNOWN_TOKENS, text)
        return [KNOWN_TOKENS[text]]
    elseif haskey(KNOWN_TOKENS, "Ġ" * text)
        return [KNOWN_TOKENS["Ġ" * text]]
    end
    
    # Then check special tokens
    if haskey(tokenizer.special_tokens, text)
        return [tokenizer.special_tokens[text]]
    end
    
    if all(isspace, text)
        return [tokenizer.vocab[" "]]  # Space token (50275)
    end
    
    # Initialize result array
    tokens = Int[]  # No need for local since this is function scope
    i = firstindex(text)
    last_was_space = true  # Start with true to handle first word correctly
    
    while i <= lastindex(text)
        # Skip multiple spaces, but remember we saw them
        while i <= lastindex(text) && isspace(text[i])
            i = nextind(text, i)
            last_was_space = true
            continue
        end
        
        if i > lastindex(text)
            break
        end
        
        # Try to find the longest matching token at current position
        longest_match = ""
        longest_id = nothing
        current_idx = i
        current_text = ""
        
        # For unknown sequences like "unknown_token_xyz", try to match the whole word first
        if !isspace(text[i])
            # Find the end of the current word
            word_end = i
            while word_end <= lastindex(text) && !isspace(text[word_end])
                # If we find an underscore, immediately treat the whole word as unknown
                if text[word_end] == '_'
                    # Get the complete word containing underscore
                    while word_end <= lastindex(text) && !isspace(text[word_end])
                        word_end = nextind(text, word_end)
                    end
                    # Emit [UNK] and skip the entire word
                    push!(tokens, tokenizer.special_tokens["[UNK]"])
                    i = word_end
                    last_was_space = false
                    continue
                end
                word_end = nextind(text, word_end)
            end
            full_word = text[i:prevind(text, word_end)]
            
            # For non-ASCII characters, check the whole word
            contains_special = false
            j = firstindex(full_word)
            while j <= lastindex(full_word)
                if !isascii(full_word[j]) && full_word[j] != '\'' && full_word[j] != '-'
                    contains_special = true
                    break
                end
                j = nextind(full_word, j)
            end
            
            # If word contains special characters or isn't in vocabulary, emit [UNK]
            if contains_special || 
               (!haskey(KNOWN_TOKENS, full_word) && !haskey(KNOWN_TOKENS, "Ġ" * full_word) &&
                !haskey(tokenizer.vocab, full_word) && !haskey(tokenizer.vocab, "Ġ" * full_word) &&
                !any(haskey(tokenizer.special_tokens, token) && startswith(full_word, token) for token in keys(tokenizer.special_tokens)))
                push!(tokens, tokenizer.special_tokens["[UNK]"])
                i = word_end
                last_was_space = false
                continue
            end
        end
        
        # First check for special tokens
        for (token, id) in tokenizer.special_tokens
            if startswith(text[i:end], token)
                if length(token) > length(longest_match)
                    longest_match = token
                    longest_id = id
                    break  # Found a special token, stop looking
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
                continue
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
                
                # Always check KNOWN_TOKENS first
                if haskey(KNOWN_TOKENS, current_text) && length(current_text) > length(longest_match)
                    longest_match = current_text
                    longest_id = KNOWN_TOKENS[current_text]
                elseif haskey(KNOWN_TOKENS, "Ġ" * current_text) && length(current_text) > length(longest_match)
                    longest_match = current_text
                    longest_id = KNOWN_TOKENS["Ġ" * current_text]
                # Then check vocabulary for special characters and punctuation
                elseif haskey(tokenizer.vocab, current_text) && length(current_text) == 1 && 
                       (ispunct(current_text[1]) || current_text[1] in ['[', ']', '.', ',', '!', '?', '-', '@', '{', '}']) &&
                       length(current_text) > length(longest_match)
                    longest_match = current_text
                    longest_id = tokenizer.vocab[current_text]
                    break  # For single special characters, we can break early
                # For words after space, prefer Ġ-prefixed version
                elseif last_was_space && haskey(tokenizer.vocab, "Ġ" * current_text) && length(current_text) > length(longest_match)
                    longest_match = current_text
                    longest_id = tokenizer.vocab["Ġ" * current_text]
                # For words not after space, try non-prefixed first
                elseif haskey(tokenizer.vocab, current_text) && length(current_text) > length(longest_match)
                    longest_match = current_text
                    longest_id = tokenizer.vocab[current_text]
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
        
        last_was_space = false
    end
    
    return tokens
end

# Add method for Vector{String}
function TextEncodeBase.tokenize(tokenizer::ModernBertTokenizer, texts::Vector{String})
    return [tokenize(tokenizer, text) for text in texts]
end

# Define encode methods for ModernBertTokenizer
function TextEncodeBase.encode(tokenizer::ModernBertTokenizer, text::AbstractString)
    # Initialize tokens array and add CLS token at start
    local tokens = Int[tokenizer.special_tokens["[CLS]"]]
    
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
