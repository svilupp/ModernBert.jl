# Define module-level constants
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
Special tokens are automatically handled during encoding and decoding.

# Examples
```julia-repl
julia> encoder = ModernBertEncoder("data/tokenizer.json")
ModernBertEncoder(...)

julia> encoder.encode("hello world")
3-element Vector{Int64}:
 15339
  1917
   264

julia> encoder.tokenizer("hello world")
["hello", "world"]
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

function Base.show(io::IO, e::ModernBertEncoder)
    print(io, "ModernBertEncoder(")
    show(io, e.tokenizer)
    print(io, ", Vocab(size = ")
    print(io, length(e.vocab))
    print(io, "))")
end
TextEncodeBase.process(e::ModernBertEncoder) = identity
function (e::ModernBertEncoder)(x::AbstractString)
    TextEncodeBase.lookup(e.vocab, encode_indices(e, x))
end

"""
    ModernBertEncoder(config_path::String)

Create a ModernBertEncoder from a configuration file path.
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

Tokenize text into tokens. If add_special_tokens is true, adds [CLS] and [SEP] tokens.
"""
function TextEncodeBase.tokenize(
        encoder::ModernBertEncoder, text::String; add_special_tokens::Bool = true)
    # Get base tokens
    tokens = encoder.tokenizer(text)

    if add_special_tokens
        # Convert to strings and add special tokens
        return ["[CLS]"; tokens; "[SEP]"]
    else
        return tokens
    end
end

"""
    encode(encoder::ModernBertEncoder, text::AbstractString)

Encode text into token IDs. If add_special_tokens is true, adds [CLS] and [SEP] token IDs.
"""
function TextEncodeBase.encode(
        encoder::ModernBertEncoder, text::AbstractString)
    # Get token IDs directly from the vocab
    ids = encoder.encode(text)
end

"""
    TextEncodeBase.encode(
        encoder::ModernBertEncoder,
        texts::AbstractVector{<:AbstractString})

Encode a vector of texts into token IDs. If pad_to_max is true, all sequences will be padded
to the length of the longest sequence using the [PAD] token ID.

Returns a matrix of token IDs where each row represents a text sequence.
"""
function TextEncodeBase.encode(
        encoder::ModernBertEncoder,
        texts::AbstractVector{<:AbstractString})
    # Encode each text individually
    encoded_sequences = [encoder.encode(text) for text in texts]
    # Find maximum length and create output matrix filled with padding tokens
    max_len = maximum(length.(encoded_sequences))
    pad_id = encoder.special_tokens["[PAD]"]
    output = fill(pad_id, max_len, length(texts))

    # Fill in the actual token ids
    for (i, seq) in enumerate(encoded_sequences)
        output[1:length(seq), i] = seq
    end

    return output
end