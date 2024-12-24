"""
    BertTextEncoder

The text encoder for Bert model (BPE tokenization).

# Fields
- `tokenizer::BPETokenizer`: The BPE tokenizer.
- `vocab::Dict{String, Int}`: The vocabulary, 0-based indexing of tokens to match Python implementation.
- `startsym::String`: The start symbol.
- `endsym::String`: The end symbol.
- `padsym::String`: The pad symbol.
- `trunc::Union{Nothing, Int}`: The truncation length. Defaults to 8192 tokens.
"""
@kwdef struct BertTextEncoder
    tokenizer::BPETokenizer
    vocab::Dict{String, Int}
    startsym::String = "[CLS]"
    endsym::String = "[SEP]"
    padsym::String = "[PAD]"
    trunc::Union{Nothing, Int} = 8192
end
function Base.show(io::IO, enc::BertTextEncoder)
    dump(io, enc; maxdepth = 1)
end

function BertTextEncoder(
        wp, vocab; startsym = "[CLS]", endsym = "[SEP]", padsym = "[PAD]", trunc = nothing)
    haskey(vocab, startsym) ||
        @warn "startsym $startsym not in vocabulary, this might cause problem."
    haskey(vocab, endsym) ||
        @warn "endsym $endsym not in vocabulary, this might cause problem."
    haskey(vocab, padsym) ||
        @warn "padsym $padsym not in vocabulary, this might cause problem."
    return BertTextEncoder(wp, vocab, startsym, endsym, padsym, trunc)
end

"""
    tokenize(enc::BertTextEncoder, text::AbstractString;
        add_special_tokens::Bool = true, add_end_token::Bool = true, token_ids::Bool = false,
        max_tokens::Union{Nothing, Int} = enc.trunc)

Tokenizes the text and returns the tokens or token IDs (to skip looking up the IDs twice).

# Arguments
- `add_special_tokens::Bool = true`: Add special tokens at the beginning and end of the text.
- `add_end_token::Bool = true`: Add end token at the end of the text.
- `token_ids::Bool = false`: If true, return the token IDs directly. Otherwise, return the tokens.
- `max_tokens::Union{Nothing, Int} = enc.trunc`: The maximum number of tokens to return (usually defined by the model).
"""
function tokenize(enc::BertTextEncoder, text::AbstractString;
        add_special_tokens::Bool = true, add_end_token::Bool = true, token_ids::Bool = false,
        max_tokens::Union{Nothing, Int} = enc.trunc)
    tokens = token_ids ? Int[] : String[]
    
    # Add start token if special tokens are requested
    if add_special_tokens
        token = token_ids ? enc.vocab[enc.startsym] : enc.startsym
        push!(tokens, token)
    end
    
    # Use BPE tokenizer to get tokens
    text_tokens = enc.tokenizer(text; token_ids=false)  # Always get string tokens first
    
    # Convert to token IDs if requested
    if token_ids
        # Map tokens to IDs using the encoder's vocabulary
        text_token_ids = Int[]
        for token in text_tokens
            # First check special tokens
            id = get(enc.tokenizer.special_tokens, token, nothing)
            if isnothing(id)
                # Then check regular vocabulary
                id = get(enc.vocab, token, enc.tokenizer.special_tokens["[UNK]"])
            end
            push!(text_token_ids, id)
        end
        text_tokens = text_token_ids
    end
    
    # Handle truncation before adding end token
    if !isnothing(enc.trunc)
        max_len = enc.trunc
        if add_special_tokens
            max_len -= 2  # Account for both special tokens
        elseif add_end_token
            max_len -= 1  # Account for end token only
        end
        if length(text_tokens) > max_len
            text_tokens = text_tokens[1:max_len]
        end
    end
    
    append!(tokens, text_tokens)
    
    # Add end token if requested
    if add_special_tokens || add_end_token
        token = token_ids ? enc.vocab[enc.endsym] : enc.endsym
        push!(tokens, token)
    end
    
    return tokens
end

"""
    encode(enc::BertTextEncoder, text::String; add_special_tokens::Bool = true,
        max_tokens::Int = enc.trunc, split_instead_trunc::Bool = false)

Encodes the text and returns the token IDs, token type IDs, and attention mask.

We enforce `max_tokens` to be a concrete number here to be able to do `split_instead_trunc`.
`split_instead_trunc` splits any long sequences into several smaller ones.
"""
function encode(enc::BertTextEncoder, text::String; add_special_tokens::Bool = true,
        max_tokens::Union{Nothing,Int} = enc.trunc, split_instead_trunc::Bool = false)
    if !split_instead_trunc
        ## Standard run - if text is longer, we truncate it and ignore
        token_ids = tokenize(enc, text; add_special_tokens, token_ids = true, max_tokens)
        # Zero indexed as models are trained for Python
        token_type_ids = zeros(Int, length(token_ids))
        attention_mask = ones(Int, length(token_ids))
    else
        ## Split run - if text is longer, we split it into multiple chunks and encode them separately
        ## Only possible with a single string to know where the chunks belong to
        ## tokenize without special tokens at first
        token_ids = tokenize(enc, text; add_special_tokens = false,
            token_ids = true, max_tokens = nothing)
        ## determine correct chunk size
        start_token = enc.vocab[enc.startsym]
        end_token = enc.vocab[enc.endsym]
        chunk_size = max_tokens - 2 * add_special_tokens
        itr = Iterators.partition(token_ids, chunk_size)
        num_chunks = length(itr)
        ## split vector in several
        mat_token_ids = zeros(Int, max_tokens, num_chunks)
        token_type_ids = zeros(Int, max_tokens, num_chunks)
        attention_mask = zeros(Int, max_tokens, num_chunks)
        @inbounds for (i, chunk) in enumerate(itr)
            if add_special_tokens
                mat_token_ids[1, i] = start_token
                attention_mask[1, i] = 1
            end
            for ri in eachindex(chunk)
                ## if special token, we shift all items by 1 down
                row_idx = add_special_tokens ? ri + 1 : ri
                mat_token_ids[row_idx, i] = chunk[ri]
                attention_mask[row_idx, i] = 1
            end
            if add_special_tokens
                row_idx = 2 + length(chunk)
                mat_token_ids[row_idx, i] = end_token
                attention_mask[row_idx, i] = 1
            end
        end
        token_ids = mat_token_ids
    end
    return token_ids, token_type_ids, attention_mask
end

# For multiple documents
function encode(enc::BertTextEncoder, passages::AbstractVector{<:AbstractString};
        add_special_tokens::Bool = true)
    tokens_vec = [tokenize(enc, passage; add_special_tokens = true, token_ids = true)
                  for passage in passages]
    max_len = maximum(length, tokens_vec) |>
              x -> isnothing(enc.trunc) ? x : min(x, enc.trunc)

    ## Assumes that padding is done with token ID 0
    token_ids = zeros(Int, max_len, length(passages))
    # Zero indexed as models are trained for Python
    token_type_ids = zeros(Int, max_len, length(passages))
    attention_mask = zeros(Int, max_len, length(passages))

    ## Encode to token IDS
    @inbounds for j in eachindex(tokens_vec)
        tokens = tokens_vec[j]
        for i in eachindex(tokens)
            if i > max_len
                break
            elseif i == max_len
                ## give [SEP] token
                token_ids[i, j] = enc.vocab[enc.endsym]
            else
                ## fill the tokens
                token_ids[i, j] = tokens[i]
            end
            attention_mask[i, j] = 1
        end
    end
    return token_ids, token_type_ids, attention_mask
end
