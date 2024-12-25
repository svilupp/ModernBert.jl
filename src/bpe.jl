"""
    BPETokenizer

A BPE (Byte-Pair Encoding) tokenizer that splits text into subword units based on learned merge rules.

# Fields
- `vocab::Dict{String,Int}`: Mapping from tokens to their IDs (0-based for Python compatibility)
- `merges::Vector{Tuple{String,String}}`: Ordered list of BPE merge rules
- `merge_index::Dict{Tuple{String,String},Int}`: Index for BPE merge rules
- `special_tokens::Dict{String,Int}`: Special token mapping (e.g., [CLS], [SEP], etc.)
- `special_token_properties::Dict{String,NamedTuple}`: Properties for special tokens (lstrip, rstrip, etc.)
- `cache::Dict{String,Vector{String}}`: Cache for already tokenized words
- `prev_token::String`: The previous token
- `add_prefix_space::Bool`: Whether to add a prefix space
- `trunc::Union{Nothing, Int}`: The truncation length. Defaults to 8192 tokens.
"""
@kwdef mutable struct BPETokenizer
    vocab::Dict{String, Int} = Dict{String, Int}()
    merges::Vector{Tuple{String, String}} = Vector{Tuple{String, String}}()
    merge_index::Dict{Tuple{String, String}, Int} = Dict{Tuple{String, String}, Int}()
    special_tokens::Dict{String, Int} = Dict{String, Int}()
    special_token_properties::Dict{String, NamedTuple} = Dict{String, NamedTuple}()
    cache::Dict{String, Vector{String}} = Dict{String, Vector{String}}()
    prev_token::String = "[CLS]"
    add_prefix_space::Bool = false
    trunc::Union{Nothing, Int} = 8192
end

"""
    load_tokenizer(config_path::String)

Load a BPE tokenizer configuration from a JSON file.

# Arguments
- `config_path::String`: Path to the tokenizer.json configuration file
"""
function load_tokenizer(config_path::String)
    @assert isfile(config_path) "Tokenizer configuration file not found: $config_path"

    config = JSON3.read(read(config_path, String))

    # Load vocabulary and merges from model configuration
    vocab = Dict{String, Int}()
    if haskey(config, "model") && haskey(config.model, "vocab")
        for (token, id) in config.model.vocab
            vocab[String(token)] = id  # IDs are already integers in the config
        end
    else
        error("No vocabulary found in tokenizer config model")
    end

    # Load merge rules preserving the exact order from config
    merges = Vector{Tuple{String, String}}()
    merge_index = Dict{Tuple{String, String}, Int}()
    if haskey(config, "model") && haskey(config.model, "merges")
        for (i, merge_rule) in enumerate(config.model.merges)
            parts = split(String(merge_rule))
            if length(parts) == 2
                # Convert parts to strings and preserve the exact order
                p1, p2 = String(parts[1]), String(parts[2])
                push!(merges, (p1, p2))
                merge_index[(p1, p2)] = i  # Add index to track merge priority
            end
        end
    else
        error("No merge rules found in tokenizer config model")
    end

    # Load special tokens and their properties
    special_tokens = Dict{String, Int}()
    special_token_properties = Dict{String, NamedTuple}()

    # Load special tokens from config
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

    ## add_prefix_space
    add_prefix_space = haskey(config, "decoder") ?
                       get(config.decoder, "add_prefix_space", false) : false

    return BPETokenizer(;
        vocab, merges, merge_index, special_tokens, special_token_properties,
        add_prefix_space)
end

"""
    get_pairs(word::Vector{String})

Get all possible pairs and longer sequences in a word represented as tokens.
Returns pairs that could potentially be merged based on the vocabulary.
"""
function get_pairs(word::Vector{String})
    pairs = Set{Tuple{String, String}}()

    # Get all consecutive pairs and potential longer sequences
    for i in 1:(length(word) - 1)
        # Basic pair
        push!(pairs, (word[i], word[i + 1]))

        # Try longer sequences
        if i < length(word) - 1
            # Try joining current token with next token
            seq = join([word[i], word[i + 1]])
            if haskey(tokenizer.vocab, seq)
                # Add pair with next token
                push!(pairs, (seq, word[i + 2]))
            end

            # Try joining next two tokens
            seq = join([word[i + 1], word[i + 2]])
            if haskey(tokenizer.vocab, seq)
                # Add pair with current token
                push!(pairs, (word[i], seq))
            end
        end
    end

    return pairs
end

# """
#     bpe_encode(tokenizer::BPETokenizer, word::AbstractString, add_prefix::Bool=true)

# Encode a single word using BPE merge operations.

# # Arguments
# - `word::AbstractString`: Word to encode (String or SubString)
# - `add_prefix::Bool`: Whether to add the 'Ġ' prefix (for word boundaries)
# """
# function bpe_encode(tokenizer::BPETokenizer, word::AbstractString,
#         add_prefix::Bool = true, token_ids::Bool = false)
#     unksym = token_ids ? tokenizer.special_tokens["[UNK]"] : "[UNK]"
#     isempty(word) && return token_ids ? Int[] : String[]

#     cache_key = add_prefix ? "Ġ" * word : word
#     if haskey(tokenizer.cache, cache_key)
#         cached_result = tokenizer.cache[cache_key]
#         if token_ids
#             return [get(tokenizer.vocab, t, unksym) for t in cached_result]
#         end
#         return copy(cached_result)
#     end

#     # Initialize with character-level tokens
#     chars = String[]
#     if add_prefix
#         push!(chars, "Ġ")
#     end

#     # Revert to splitting non-ASCII codepoints into raw UTF-8 bytes:
#     for c in word
#         push_char!(chars, c)
#     end

#     result = chars

#     # Apply merge rules until no more merges possible
#     while true
#         valid_merges = Tuple{Int, Int, Int}[]

#         # Find all possible merges
#         for i in 1:(length(result) - 1)
#             pair = (result[i], result[i + 1])
#             if haskey(tokenizer.merge_index, pair)
#                 push!(valid_merges, (i, i + 1, tokenizer.merge_index[pair]))
#             end
#         end

#         isempty(valid_merges) && break

#         # Apply highest priority merge
#         sort!(valid_merges, by = x -> x[3])
#         start_idx, end_idx, _ = valid_merges[1]
#         merged = join(result[start_idx:end_idx])

#         # Update result with merged token
#         result = vcat(result[1:(start_idx - 1)], [merged], result[(end_idx + 1):end])
#     end

#     # Cache result if valid
#     if !isempty(result) && all(t -> haskey(tokenizer.vocab, t), result)
#         tokenizer.cache[cache_key] = copy(result)
#     end

#     # Convert to token IDs if requested
#     if token_ids
#         return [get(tokenizer.vocab, t, unksym) for t in result]
#     end
#     return result
# end

# Local fallback for isalnum, checking ASCII letters or digits
function isalnum(c::Char)::Bool
    return isletter(c) || isdigit(c)
end
# We'll consider "in-word punctuation" the case where:
#   1) there's no whitespace around it,
#   2) and it's part of "domain-like" strings such as .com or john@doe.org.
function is_inword_punct(c::Char, prev_in_word::Bool, prev_char, next_char)
    # Apostrophes or hyphens mid-word, e.g. O'Neill, co-workers
    if (c == '\'' || c == '-') && prev_in_word
        return true
    end
    # For domain-like strings "ABC.com" or "john@doe.org":
    #   if there's a letter/digit on both sides with no space,
    #   consider it "in-word".
    if c in ('.', '@') &&
       prev_char !== nothing && isalnum(prev_char) &&
       next_char !== nothing && isalnum(next_char)
        return true
    end
    return false
end

# Helper to push one character as ASCII or split it into raw UTF-8 bytes if it's non-ASCII
function push_char!(chars::Vector{String}, c::Char, tokenizer::BPETokenizer)
    str = string(c)

    # For ASCII characters
    if codepoint(c) <= 0x7f
        push!(chars, str)
        return
    end

    # For Unicode characters, try to find the best match in vocabulary
    # First try the character as-is
    if haskey(tokenizer.vocab, str)
        push!(chars, str)
        return
    end

    # If not in vocab, push as unknown token
    push!(chars, "[UNK]")
end

"""
    tokenize(tokenizer::BPETokenizer, text::AbstractString;
        token_ids::Bool = false, add_special_tokens::Bool = true,
        add_prefix_space::Bool = tokenizer.add_prefix_space)

Function call overloading for BPETokenizer to make it callable directly.
Tokenizes text using the BPE algorithm.

# Arguments
- `text::AbstractString`: Input text to tokenize
- `token_ids::Bool=false`: If true, return token IDs instead of tokens
- `add_special_tokens::Bool=true`: Whether to add special tokens
- `add_prefix_space::Bool=false`: Whether the first token should have a word boundary marker

# Returns
- Vector of tokens or token IDs
"""
function tokenize(tokenizer::BPETokenizer, text::AbstractString;
        token_ids::Bool = false, add_special_tokens::Bool = true,
        add_prefix_space::Bool = tokenizer.add_prefix_space)
    ## Prepare key tokens
    startsym = token_ids ? tokenizer.special_tokens["[CLS]"] : "[CLS]"
    endsym = token_ids ? tokenizer.special_tokens["[SEP]"] : "[SEP]"
    unksym = token_ids ? tokenizer.special_tokens["[UNK]"] : "[UNK]"

    # Initialize result vector with correct type
    tokens = token_ids ? Vector{Int}() : Vector{String}()

    # Add [CLS] token if requested
    if add_special_tokens
        push!(tokens, startsym)
    end

    # Handle empty input - but continue to add special tokens if needed
    if isempty(text)
        if add_special_tokens
            push!(tokens, endsym)
        end
        return tokens
    end

    # Get list of special tokens for preservation
    special_token_list = collect(keys(tokenizer.special_tokens))

    # Initialize character sequence
    chars = String[]
    current_pos = firstindex(text)
    text_length = lastindex(text)
    in_word = false  # Track if we're inside a word
    need_boundary = add_prefix_space  # Initialize based on add_prefix_space

    while current_pos <= text_length
        # Check for special tokens first
        found_special = false
        for special_token in special_token_list
            if startswith(@view(text[current_pos:text_length]), special_token)
                push!(chars, special_token)
                for _ in 1:length(special_token)
                    current_pos = nextind(text, current_pos)
                end
                found_special = true
                in_word = false
                need_boundary = true
                break
            end
        end
        found_special && continue

        # Get current character
        c = text[current_pos]

        # Handle whitespace
        if isspace(c)
            # Try to match with potential merges first
            remaining_text = @view(text[current_pos:text_length])
            found_merge = false

            # Get all possible whitespace tokens and sort by length (longest first)
            whitespace_tokens = [(token, id)
                                 for (token, id) in tokenizer.vocab if all(isspace, token)]
            sort!(whitespace_tokens, by = x -> length(x[1]), rev = true)

            for (token, _) in whitespace_tokens
                if startswith(remaining_text, token)
                    push!(chars, token)
                    current_pos = nextind(text, current_pos, length(token))
                    found_merge = true
                    break
                end
            end

            if !found_merge
                in_word = false
                need_boundary = true
                current_pos = nextind(text, current_pos)
            end
            continue
        end

        # Determine if the character is "in-word" punctuation or a true word boundary
        next_char = current_pos < text_length ? text[nextind(text, current_pos)] : nothing

        # Check if this punctuation is in-word or truly breaks words
        if ispunct(c)
            prev_char = current_pos > firstindex(text) ? text[prevind(text, current_pos)] :
                        nothing
            c_inword = is_inword_punct(c, in_word, prev_char, next_char)

            if c_inword
                push_char!(chars, c, tokenizer)
                in_word = true
                current_pos = nextind(text, current_pos)
                continue
            end

            # If we already needed a boundary before punctuation, insert it
            if need_boundary
                push!(chars, "Ġ")
                need_boundary = false
            end
            # Now push the punctuation itself exactly once
            push_char!(chars, c, tokenizer)
            in_word = false
            # Decide whether we add a boundary for the next token:
            #   1) If there's nothing left or if the next character is whitespace, need_boundary = true
            #   2) Otherwise, keep it false so the next token doesn't get "Ġ" (e.g. "[and]" → "[", "and")
            if next_char === nothing || (next_char != nothing && isspace(next_char))
                need_boundary = true
            else
                need_boundary = false
            end
            current_pos = nextind(text, current_pos)
            continue
        end

        # Special cases for punctuation that shouldn't break words
        if ispunct(c) && (c == '\'' || c == '-') && in_word
            push_char!(chars, c, tokenizer)
            current_pos = nextind(text, current_pos)
            continue
        end

        # If we're about to start a new word and need_boundary is set, add a boundary
        if need_boundary && !in_word
            push!(chars, "Ġ")
            need_boundary = false
        end

        push_char!(chars, c, tokenizer)
        in_word = true
        current_pos = nextind(text, current_pos)
    end

    # Apply BPE merges to entire sequence
    result = chars
    while true
        valid_merges = Tuple{Int, Int, Int}[]

        # Find all possible merges across the entire sequence
        for i in 1:(length(result) - 1)
            pair = (result[i], result[i + 1])
            if haskey(tokenizer.merge_index, pair)
                push!(valid_merges, (i, i + 1, tokenizer.merge_index[pair]))
            end
        end

        isempty(valid_merges) && break

        # Apply highest priority merge
        sort!(valid_merges, by = x -> x[3])
        start_idx, end_idx, _ = valid_merges[1]
        merged = join(result[start_idx:end_idx])

        # Update result with merged token
        result = vcat(result[1:(start_idx - 1)], [merged], result[(end_idx + 1):end])
    end

    # Convert to token IDs if requested and add to final tokens
    for token in result
        if token_ids
            id = get(tokenizer.vocab, token, unksym)
            push!(tokens, id)
        else
            push!(tokens, token)
        end
    end

    # Add [SEP] token if requested
    if add_special_tokens
        push!(tokens, endsym)
    end

    return tokens
end

# Function call operator
function (tokenizer::BPETokenizer)(
        text::AbstractString; token_ids::Bool = false, add_special_tokens::Bool = true,
        add_prefix_space::Bool = tokenizer.add_prefix_space)
    return tokenize(tokenizer, text; token_ids, add_special_tokens, add_prefix_space)
end

"""
    encode(tokenizer::BPETokenizer, text::String; add_special_tokens::Bool = true)

Encodes the text and returns the token IDs, token type IDs, and attention mask.
"""
function encode(tokenizer::BPETokenizer, text::String; add_special_tokens::Bool = true)
    # Get token IDs using the tokenizer
    token_ids = tokenize(tokenizer, text;
        add_special_tokens = add_special_tokens,
        token_ids = true
    )

    # Zero indexed as models are trained for Python
    token_type_ids = zeros(Int, length(token_ids))
    attention_mask = ones(Int, length(token_ids))
    return token_ids, token_type_ids, attention_mask
end

# For multiple documents
function encode(tokenizer::BPETokenizer, passages::AbstractVector{<:AbstractString};
        add_special_tokens::Bool = true)
    # Tokenize each passage
    tokens_vec = [tokenize(tokenizer, passage;
                      add_special_tokens = add_special_tokens,
                      token_ids = true
                  ) for passage in passages]
    max_len = maximum(length, tokens_vec)

    # Assumes that padding is done with token ID of [PAD]
    token_ids = fill(tokenizer.special_tokens["[PAD]"], max_len, length(passages))
    token_type_ids = zeros(Int, max_len, length(passages))
    attention_mask = zeros(Int, max_len, length(passages))

    # Fill token IDs and attention mask
    @inbounds for j in eachindex(tokens_vec)
        tokens = tokens_vec[j]
        for i in eachindex(tokens)
            token_ids[i, j] = tokens[i]
            attention_mask[i, j] = 1
        end
    end
    return token_ids, token_type_ids, attention_mask
end