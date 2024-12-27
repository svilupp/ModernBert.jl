# Define module-level constants
# Helper function to detect emoji and other special Unicode characters
function is_special_unicode(c::Char)
    cat = Base.Unicode.category_code(c)
    # Unicode categories for emoji and symbols
    return (cat == 28) ||  # So (Symbol_other)
           (cat == 25) ||  # Sc (Symbol_currency)
           (cat == 24) ||  # Sm (Symbol_math)
           (0x1F300 ≤ UInt32(c) ≤ 0x1F9FF)  # Emoji block
end

const DEFAULT_SPECIAL_TOKENS = Dict{String, Int}(
    "[UNK]" => 50280,
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[PAD]" => 50283,
    "[MASK]" => 50284
)

"""
    ModernBertEncoder

A wrapper around ModernBertTokenizer that provides encode/decode functionality compatible with BytePairEncoding.jl.
Special tokens ([CLS], [SEP], [MASK], [PAD], [UNK]) are automatically handled during encoding and decoding.

# Special Token IDs
- [UNK]: 50280 - Used for unknown tokens
- [CLS]: 50281 - Added at the start of sequences
- [SEP]: 50282 - Added at the end of sequences
- [PAD]: 50283 - Used for padding in batch operations
- [MASK]: 50284 - Used for masked language modeling

# Examples
```julia-repl
julia> encoder = ModernBertEncoder("data/tokenizer.json")
ModernBertEncoder(...)

# Tokenization (returns string tokens)
julia> tokenize(encoder, "hello world")
4-element Vector{String}:
 "[CLS]"
 "hello"
 "world"
 "[SEP]"

# Encoding (returns token IDs)
julia> encode(encoder, "hello world")
4-element Vector{Int64}:
 50281  # [CLS]
 15339  # hello
  1917  # world
 50282  # [SEP]

# Batch encoding (returns matrix of token IDs)
julia> encode(encoder, ["hello", "world"])
4×2 Matrix{Int64}:
 50281  50281  # [CLS]
 15339   1917  # tokens
 50282  50282  # [SEP]
 50283  50283  # [PAD]
```
"""
struct ModernBertEncoder <: TextEncodeBase.AbstractTextEncoder
    tokenizer::Any  # BytePairEncoding tokenizer
    vocab::Vocab
    special_tokens::Dict{String, Int}
    id_to_token::Dict{Int, String}
end
function Base.propertynames(e::ModernBertEncoder)
    (:encode, :decode, fieldnames(ModernBertEncoder)...)
end
function Base.getproperty(e::ModernBertEncoder, sym::Symbol)
    if sym == :encode
        return e
    elseif sym == :decode
        return Base.Fix1(TextEncodeBase.decode_text, e)
    else
        return getfield(e, sym)
    end
end

"""
    Base.show(io::IO, e::ModernBertEncoder)

Pretty-print a ModernBertEncoder instance, showing the underlying tokenizer and vocabulary size.
"""
function Base.show(io::IO, e::ModernBertEncoder)
    print(io, "ModernBertEncoder(")
    show(io, e.tokenizer)
    print(io, ", Vocab(size = ")
    print(io, length(e.vocab))
    print(io, "))")
end
"""
    TextEncodeBase.process(e::ModernBertEncoder)

Process function for TextEncodeBase compatibility. Returns the identity function as no
pre-processing is needed for ModernBertEncoder.
"""
TextEncodeBase.process(e::ModernBertEncoder) = identity
function (e::ModernBertEncoder)(x::AbstractString)
    TextEncodeBase.lookup(e.vocab, encode_indices(e, x))
end

"""
    ModernBertEncoder(config_path::String)

Create a ModernBertEncoder from a configuration file path. The configuration file should be
a JSON file containing the vocabulary and merge rules for the tokenizer.

The encoder validates that all special tokens ([CLS], [SEP], [MASK], [PAD], [UNK]) are present
in the vocabulary with the correct IDs. It also initializes the BPE tokenizer with the
merge rules from the configuration.

# Arguments
- `config_path::String`: Path to the tokenizer configuration JSON file

# Returns
- `ModernBertEncoder`: A new encoder instance

# Throws
- `AssertionError`: If the config file is not found
- `ErrorException`: If special tokens are missing or have incorrect IDs
"""
function ModernBertEncoder(config_path::String)
    @assert isfile(config_path) "Config file not found at $config_path"
    config = JSON3.read(read(config_path, String))

    # Use module-level special tokens mapping
    special_tokens = copy(DEFAULT_SPECIAL_TOKENS)

    # Create vocabulary mapping
    vocab_dict = Dict{String, Int}()

    # Load vocabulary with exact GPT2 token mappings
    for (token, id) in config.model.vocab
        token_str = String(token)
        token_id = parse(Int, string(id))
        vocab_dict[token_str] = token_id
    end

    # Load additional tokens if present
    if haskey(config, :added_tokens)
        for token in config.added_tokens
            vocab_dict[String(token.content)] = token.id
        end
    end

    # Validate special tokens
    for (token, expected_id) in DEFAULT_SPECIAL_TOKENS
        id = get(vocab_dict, token, nothing)
        id == nothing && error("Special token $token not found in vocabulary")
        id != expected_id &&
            error("Special token $token has incorrect ID: expected $expected_id, got $id")
    end

    # Initialize BPE tokenizer
    bpe_merges = Dict{Tuple{Merge, Merge}, Int}()
    for (i, merge_entry) in enumerate(config.model.merges)
        try
            pair = parse_merge(string(merge_entry))
            bpe_merges[pair] = i
        catch e
            @warn "Skipping invalid merge rule: $merge_entry"
        end
    end

    # Create tokenizer pipeline
    base_tokenizer = BPE(bpe_merges)
    tokenizer = BPETokenizer(
        CodeNormalizer(
        BPETokenization(
            GPT2Tokenization(),
            base_tokenizer
        ),
        gpt2_codemap()
    )
    )

    # Create reverse mapping
    id_to_token = Dict{Int, String}(id => token for (token, id) in vocab_dict)
    merge!(id_to_token, Dict(id => token for (token, id) in special_tokens))

    # Create lookup vector for Vocab
    max_id = maximum(values(vocab_dict))
    vocab_lookup = fill("", 1 + max_id)
    for (token, id) in vocab_dict
        vocab_lookup[1 + id] = token
    end

    vector = PerforatedOverwritableLookupVector(
        DATLookupVector(vocab_lookup),
        DictBackedLookupDict(
            special_tokens, Dict(v => k for (k, v) in special_tokens)))

    vocab = Vocab(vector, "", 0)

    ModernBertEncoder(tokenizer, vocab, special_tokens, id_to_token)
end

"""
    tokenize(encoder::ModernBertEncoder, text::String; add_special_tokens::Bool=true)

Tokenize text into tokens using byte-pair encoding (BPE). Special tokens are preserved during
tokenization, and if `add_special_tokens` is true (default), [CLS] and [SEP] tokens are added
at the start and end of the sequence respectively.

# Arguments
- `encoder::ModernBertEncoder`: The encoder instance
- `text::String`: The text to tokenize
- `add_special_tokens::Bool=true`: Whether to add [CLS] and [SEP] tokens

# Returns
- `Vector{String}`: A vector of tokens

# Examples
```julia-repl
julia> encoder = ModernBertEncoder("tokenizer.json")

# Basic tokenization with special tokens
julia> tokenize(encoder, "hello world")
4-element Vector{String}:
 "[CLS]"
 "hello"
 "world"
 "[SEP]"

# Tokenization without special tokens
julia> tokenize(encoder, "hello world", add_special_tokens=false)
2-element Vector{String}:
 "hello"
 "world"

# Special token preservation
julia> tokenize(encoder, "This is a [MASK] test")
6-element Vector{String}:
 "[CLS]"
 "This"
 "is"
 "a"
 "[MASK]"
 "test"
 "[SEP]"
```
"""
function TextEncodeBase.tokenize(
        encoder::ModernBertEncoder, text::String; add_special_tokens::Bool = true)
    # Split text into parts while preserving special tokens
    parts = String[]
    current = ""
    i = firstindex(text)
    
    # Sort special tokens by length (longest first) to handle overlapping tokens correctly
    special_tokens = sort(collect(keys(encoder.special_tokens)), by=length, rev=true)
    
    while i ≤ ncodeunits(text)
        # Check for special tokens
        found_special = false
        rest_of_text = @view(text[i:end])
        for token in special_tokens
            if startswith(rest_of_text, token)
                # Add accumulated text if any
                if !isempty(current)
                    push!(parts, current)
                    current = ""
                end
                push!(parts, token)
                i += ncodeunits(token)
                found_special = true
                break
            end
        end
        if !found_special
            # Handle Unicode characters properly
            current *= string(text[i])
            i = nextind(text, i)
        end
    end
    
    # Add any remaining text
    if !isempty(current)
        push!(parts, current)
    end

    # Process each part
    tokens = String[]
    for part in parts
        if part in special_tokens
            push!(tokens, part)
        else
            # Check for special Unicode characters first
            if any(is_special_unicode, part)
                push!(tokens, "[UNK]")
                continue
            end
            
            # Handle potential unknown tokens
            local part_tokens
            try
                part_tokens = encoder.tokenizer(part)
            catch
                push!(tokens, "[UNK]")
                continue
            end
            
            # Check if tokenization produced valid tokens
            if isempty(part_tokens)
                push!(tokens, "[UNK]")
            else
                # Check if all tokens are valid (in vocabulary and not too long)
                valid_tokens = true
                max_token_length = 64  # Reasonable maximum token length
                for t in part_tokens
                    # Skip extremely long tokens
                    if length(t) > max_token_length
                        valid_tokens = false
                        break
                    end
                    # Check if token exists in vocabulary
                    if !haskey(encoder.special_tokens, t)
                        try
                            # Use direct vocabulary lookup
                            if !haskey(encoder.vocab, t)
                                valid_tokens = false
                                break
                            end
                        catch
                            valid_tokens = false
                            break
                        end
                    end
                end
                
                if valid_tokens
                    append!(tokens, part_tokens)
                else
                    push!(tokens, "[UNK]")
                end
            end
        end
    end

    if add_special_tokens
        return ["[CLS]"; tokens; "[SEP]"]
    else
        return tokens
    end
end

"""
    encode(encoder::ModernBertEncoder, text::AbstractString)

Encode text into token IDs. Automatically adds [CLS] and [SEP] token IDs.

# Examples
```julia-repl
julia> encoder = ModernBertEncoder("tokenizer.json")
julia> encode(encoder, "hello world")
3-element Vector{Int64}:
 50281  # [CLS]
 15339  # hello
  1917  # world
 50282  # [SEP]
```
"""
function TextEncodeBase.encode(
        encoder::ModernBertEncoder, text::AbstractString)
    # Get tokens first (includes [CLS] and [SEP])
    tokens = tokenize(encoder, text)
    
    # Convert tokens to IDs
    token_ids = Int[]
    for token in tokens
        if token in keys(encoder.special_tokens)
            push!(token_ids, encoder.special_tokens[token])
        else
            # Handle regular tokens and unknown tokens
            if haskey(encoder.vocab, token)
                push!(token_ids, encoder.vocab[token])
            else
                push!(token_ids, encoder.special_tokens["[UNK]"])
            end
        end
    end
    
    return token_ids
end

"""
    TextEncodeBase.encode(
        encoder::ModernBertEncoder,
        texts::AbstractVector{<:AbstractString})

Encode a vector of texts into token IDs. Automatically pads sequences to the length of the
longest sequence using the [PAD] token ID and adds [CLS] and [SEP] tokens.

Returns a matrix of token IDs where each column represents a text sequence.

# Examples
```julia-repl
julia> encoder = ModernBertEncoder("tokenizer.json")
julia> encode(encoder, ["hello", "world"])
4×2 Matrix{Int64}:
 50281  50281  # [CLS]
 15339   1917  # hello, world
 50282  50282  # [SEP]
 50283  50283  # [PAD]
```
"""
function TextEncodeBase.encode(
        encoder::ModernBertEncoder,
        texts::AbstractVector{<:AbstractString})
    # Encode each text individually
    encoded_sequences = [encode(encoder, text) for text in texts]
    
    # Find maximum length
    max_len = maximum(length.(encoded_sequences))
    pad_id = encoder.special_tokens["[PAD]"]
    
    # Create output matrix filled with padding tokens
    output = fill(pad_id, max_len, length(texts))
    
    # Fill in the actual token ids
    for (i, seq) in enumerate(encoded_sequences)
        output[1:length(seq), i] = seq
    end
    
    return output
end
