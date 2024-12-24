## This code is forked from Transformers.jl: https://github.com/chengchingwen/Transformers.jl

using Unicode: normalize  # Add explicit import for normalize function
using JSON3

export Tokenizer, tokenize, create_tokenizer, add_special_tokens, add_end_token

include("bpe.jl")  # Include our BPE implementation

"""
    Tokenizer

Main tokenizer struct that wraps the BPE tokenizer implementation.

# Fields
- `bpe::BPETokenizer`: The underlying BPE tokenizer
"""
struct Tokenizer
    bpe::BPETokenizer
end

"""
    create_tokenizer(config_path::String)

Create a new tokenizer from a configuration file.

# Arguments
- `config_path::String`: Path to the tokenizer.json configuration file
"""
function create_tokenizer(config_path::String)
    if !isfile(config_path)
        error("Tokenizer configuration file not found: $config_path")
    end
    bpe = create_bpe_tokenizer(config_path)
    return Tokenizer(bpe)
end

"""
    tokenize(tokenizer::Tokenizer, text::AbstractString; token_ids::Bool=false)

Tokenize text using the BPE tokenizer.

# Arguments
- `tokenizer::Tokenizer`: The tokenizer to use
- `text::AbstractString`: Input text to tokenize
- `token_ids::Bool=false`: If true, return token IDs instead of tokens

# Returns
- Vector of tokens or token IDs
"""
function tokenize(tokenizer::Tokenizer, text::AbstractString; token_ids::Bool=false, add_special_tokens::Bool=true)
    # Get base tokens
    tokens = tokenizer.bpe(text; token_ids=token_ids)
    
    # Add special tokens if requested
    if add_special_tokens
        if token_ids
            cls_id = tokenizer.bpe.special_tokens["[CLS]"]
            sep_id = tokenizer.bpe.special_tokens["[SEP]"]
            return vcat([cls_id], tokens, [sep_id])
        else
            return vcat(["[CLS]"], tokens, ["[SEP]"])
        end
    end
    
    return tokens
end

function isinvalid(c)
    if c == '\t' || c == '\n' || c == '\r'
        return false
    end
    c == Char(0) || c == Char(0xfffd) || iscntrl(c)
end

# ignore invalid characters such like U+0000, U+fffd, and Control characters
function invalid(ts)
    isinvalid(ts[]) || return false
    ts.idx += 1
    return true
end

function ischinese(c)
    Char(0x4e00) ≤ c ≤ Char(0x9fff) ||
        Char(0x3400) ≤ c ≤ Char(0x4dbf) ||
        Char(0x20000) ≤ c ≤ Char(0x2a6df) ||
        Char(0x2a700) ≤ c ≤ Char(0x2b73f) ||
        Char(0x2b740) ≤ c ≤ Char(0x2b81f) ||
        Char(0x2b820) ≤ c ≤ Char(0x2ceaf) ||
        Char(0xf900) ≤ c ≤ Char(0xfaff) ||
        Char(0x2f800) ≤ c ≤ Char(0x2fa1f)
end

# separate on chinese characters
function chinese(ts)
    ischinese(ts[]) || return false
    flush!(ts, string(ts[]))
    ts.idx += 1
    return true
end

function isbertpunct(c)
    ispunct(c) ||
        Char(33) ≤ c ≤ Char(47) ||
        Char(58) ≤ c ≤ Char(64) ||
        Char(91) ≤ c ≤ Char(96) ||
        Char(123) ≤ c ≤ Char(126)
end

function bertpunct(ts)
    isbertpunct(ts[]) || return false
    flush!(ts, string(ts[]))
    ts.idx += 1
    return true
end

iscatemn(c) = Base.Unicode.category_code(c) == Base.Unicode.UTF8PROC_CATEGORY_MN
function catemn(ts)
    iscatemn(ts[]) || return false
    ts.idx += 1
    return true
end

#=
bert basic tokenizer pipeline
skip 1. convert to unicode
2. clean text
3. handle chinese character
4. tokenize with white space
5. if lower case : lower, NFD normalize, skip 'Mn' unicode on each tokens
6. split each token with punct and punct remain

=#
function _bert_tokenise(input, ::Val{lower}, special_tokens=String[]) where {lower}
    # First check for special tokens and preserve them
    tokens = String[]
    current_pos = 1
    input_len = length(input)
    
    while current_pos <= input_len
        # Check if any special token starts at current position
        found_special = false
        for special in special_tokens
            if startswith(SubString(input, current_pos), special)
                push!(tokens, special)
                current_pos += length(special)
                found_special = true
                break
            end
        end
        
        if !found_special
            # Process the next segment until potential special token
            next_special = input_len + 1
            for special in special_tokens
                pos = findnext(special, input, current_pos)
                if !isnothing(pos)
                    # If pos is a range, take its start
                    pos_start = pos isa Integer ? pos : first(pos)
                    next_special = min(next_special, pos_start)
                end
            end
            
            # Tokenize the segment before next special token
            end_pos = next_special - 1
            if end_pos >= current_pos
                segment = SubString(input, current_pos, end_pos)
                if !isempty(segment)
                    ts = TokenBuffer(lower ? normalize(lowercase(String(segment)), :NFD) : String(segment))
                    while !isdone(ts)
                        (lower && catemn(ts)) ||
                            invalid(ts) ||
                            chinese(ts) ||
                            spaces(ts) ||
                            bertpunct(ts) ||
                            character(ts)
                    end
                    append!(tokens, ts.tokens)
                end
            end
            current_pos = next_special
        end
    end
    return tokens
end

"""
    bert_uncased_tokenizer(input)

Google bert tokenizer which do lower case on input before tokenization.
"""
bert_uncased_tokenizer(input) = _bert_tokenise(input, Val(true))

"""
    bert_cased_tokenizer(input)

Google bert tokenizer which remain the case during tokenization. Recommended for multi-lingual data.
"""
bert_cased_tokenizer(input) = _bert_tokenise(input, Val(false))

function extract_added_token(added_token)
    vidx = added_token["id"] + 1
    token = added_token["content"]
    isspecial = added_token["special"]

    added_token["rstrip"] ||
        added_token["lstrip"] && tokenizer_warn(
            "match token `$token` require to match with space on either side but that is not implemented here"
        )
    added_token["single_word"] && tokenizer_warn(
        "match token `$token` does not match inside of a word but that is not implemented here"
    )
    return vidx, token, isspecial
end

extract_and_add_tokens!(::Nothing, _) = nothing
function extract_and_add_tokens!(added_token_list, vocab_list)
    iszero(length(added_token_list)) && return nothing
    added_token_list = sort(added_token_list; by = Base.Fix2(getindex, "id"))
    match_tokens = String[]
    for added_token in added_token_list
        vidx, token, isspecial = extract_added_token(added_token)
        if isspecial
            if vidx > length(vocab_list)
                # special tokens not in the vocab already
                @assert vidx == length(vocab_list) + 1
                push!(vocab_list, token)
            end
            @assert vocab_list[vidx] == token
        else
            n_vocab = length(vocab_list)
            if vidx == n_vocab + 1
                push!(vocab_list, token)
            elseif vidx <= n_vocab
                @assert vocab_list[vidx]==token "Two word has same index: $(token) and $(vocab_list[idx])"
            else
                error("There is a gap in the vocabulary")
            end
        end
        push!(match_tokens, token)
    end
    return match_tokens
end

function reverse_keymap_to_list(dict)
    vocab_list = Vector{String}(undef, length(dict))
    for (k, v) in dict
        v += 1
        @assert !isassigned(vocab_list, v) "Two word has same index: $(k) and $(vocab_list[v])"
        vocab_list[v] = String(k)
    end
    @assert all(Base.Fix1(isassigned, vocab_list), eachindex(vocab_list)) "There is a gap in the vocabulary"
    return vocab_list
end

mutable struct TokenBuffer
    text::String
    tokens::Vector{String}
    buffer::IOBuffer
    idx::Int
end

function TokenBuffer(text::String)
    return TokenBuffer(text, String[], IOBuffer(), 1)
end

Base.getindex(ts::TokenBuffer) = ts.text[ts.idx]
Base.length(ts::TokenBuffer) = length(ts.text)
isdone(ts::TokenBuffer) = ts.idx > length(ts)

function flush!(ts::TokenBuffer, s = "")
    if ts.buffer.size > 0
        push!(ts.tokens, String(take!(ts.buffer)))
    end
    if !isempty(s)
        push!(ts.tokens, s)
    end
    return ts
end

function spaces(ts::TokenBuffer)
    isspace(ts[]) || return false
    ts.idx += 1
    flush!(ts)
    while !isdone(ts) && isspace(ts[])
        ts.idx += 1
    end
    return true
end

function character(ts::TokenBuffer)
    write(ts.buffer, ts[])
    ts.idx += 1
    return true
end

function tokenizer_warn(msg)
    @warn msg
end
