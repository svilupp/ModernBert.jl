module ModernBertTokenizerImpl

export ModernBertTokenizer, load_modernbert_tokenizer, tokenize, encode

using Unicode
using TextEncodeBase
using BytePairEncoding
using JSON3

import TextEncodeBase: encode, tokenize, FlatTokenizer, CodeNormalizer, Sentence, getvalue
import BytePairEncoding: BPE, GPT2Tokenization, BPETokenization, gpt2_codemap

# Special token IDs that must be preserved exactly
const SPECIAL_TOKEN_IDS = Dict{String, Int}(
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[MASK]" => 50284,
    "[PAD]" => 50283,
    "[UNK]" => 50280,
    # Additional padding tokens to complete vocabulary
    "[PAD1]" => 50285,
    "[PAD2]" => 50286,
    "[PAD3]" => 50287
)

# Basic tokenizer structure
mutable struct ModernBertTokenizer
    tokenizer::FlatTokenizer
    vocab::Dict{String, Int}
    id_to_token::Dict{Int, String}
    special_tokens::Dict{String, Int}
    cache::Dict{String, Vector{Int}}
end

function load_modernbert_tokenizer(config_path::String)
    @assert isfile(config_path) "Tokenizer configuration file not found: $config_path"
    
    # Load configuration
    config = JSON3.read(read(config_path, String))
    
    # Create vocabulary mappings
    vocab = Dict{String, Int}()
    id_to_token = Dict{Int, String}()
    
    # Add special tokens first to ensure exact IDs
    for (token, id) in SPECIAL_TOKEN_IDS
        vocab[token] = id
        id_to_token[id] = token
    end
    
    # Add special tokens first to ensure exact IDs
    for (token, id) in SPECIAL_TOKEN_IDS
        vocab[token] = id
        id_to_token[id] = token
    end
    
    # Add vocabulary from config
    for (token, id) in config.model.vocab
        token = String(token)
        # Skip if token is already in vocab (special tokens)
        if !haskey(vocab, token)
            vocab[token] = id
            id_to_token[id] = token
        end
    end
    
    # Create merge rules
    merges = Dict{NTuple{2, BytePairEncoding.Merge}, Int}()
    for (i, merge_rule) in enumerate(config.model.merges)
        parts = split(String(merge_rule))
        if length(parts) == 2
            merge_pair = (BytePairEncoding.Merge(parts[1]), BytePairEncoding.Merge(parts[2]))
            merges[merge_pair] = i
        end
    end
    
    # Create tokenizer pipeline following BytePairEncoding.jl's approach
    base_tokenizer = GPT2Tokenization()  # Handles byte-level pre-tokenization
    bpe = BPE(merges)  # BPE with merge rules
    
    # Create tokenizer pipeline with special token handling
    base_tkr = BPETokenization(base_tokenizer, bpe)
    normalized_tkr = TextEncodeBase.CodeNormalizer(base_tkr, gpt2_codemap())
    tokenizer = FlatTokenizer(normalized_tkr)
    
    # Initialize with empty cache
    ModernBertTokenizer(tokenizer, vocab, id_to_token, SPECIAL_TOKEN_IDS, Dict{String, Vector{Int}}())
end



function TextEncodeBase.tokenize(tokenizer::ModernBertTokenizer, text::AbstractString; token_ids::Bool=true, add_special_tokens::Bool=false)
    # Handle empty text
    if isempty(text)
        if add_special_tokens
            return token_ids ? 
                [tokenizer.special_tokens["[CLS]"], tokenizer.special_tokens["[SEP]"]] :
                ["[CLS]", "[SEP]"]
        else
            return token_ids ? Int[] : String[]
        end
    end
    
    # Handle special tokens directly
    if haskey(tokenizer.special_tokens, text)
        return token_ids ? [tokenizer.special_tokens[text]] : [text]
    end
    
    # Check cache for exact match
    cache_key = "$(text)_$(add_special_tokens)"
    if token_ids && haskey(tokenizer.cache, cache_key)
        return copy(tokenizer.cache[cache_key])
    end
    
    # Process text
    result = token_ids ? Int[] : String[]
    
    # Add [CLS] if needed
    if add_special_tokens
        if token_ids
            push!(result, tokenizer.special_tokens["[CLS]"])
        else
            push!(result, "[CLS]")
        end
    end
    
    # Split text into parts, preserving special tokens
    parts = String[]
    current = ""
    i = firstindex(text)
    
    while i <= lastindex(text)
        # Check for special tokens
        found_special = false
        for special_token in sort(collect(keys(tokenizer.special_tokens)), by=length, rev=true)
            token_length = length(special_token)
            end_idx = i
            valid_match = true
            
            # Use nextind to safely traverse the string
            for _ in 1:token_length-1
                if end_idx > lastindex(text)
                    valid_match = false
                    break
                end
                end_idx = nextind(text, end_idx)
            end
            
            if valid_match && end_idx <= lastindex(text)
                potential_match = text[i:end_idx]
                if potential_match == special_token
                    # Add accumulated text if any
                    if !isempty(current)
                        push!(parts, current)
                        current = ""
                    end
                    # Add the special token
                    push!(parts, special_token)
                    i = nextind(text, end_idx)
                    found_special = true
                    break
                end
            end
        end
        
        if !found_special
            current *= text[i]
            i = nextind(text, i)
        end
    end
    
    # Add any remaining text
    if !isempty(current)
        push!(parts, current)
    end
    
    # Process each part
    for part in parts
        if haskey(tokenizer.special_tokens, part)
            # Handle special tokens directly
            if token_ids
                push!(result, tokenizer.special_tokens[part])
            else
                push!(result, part)
            end
        else
            # Skip empty parts
            isempty(strip(part)) && continue
            
            # Tokenize non-special text
            sentence = Sentence(part)
            tokens = tokenizer.tokenizer(sentence)
            
            for token in tokens
                token_str = String(getvalue(token))
                if token_ids
                    # Look up token ID, fallback to UNK
                    token_id = get(tokenizer.vocab, token_str, tokenizer.special_tokens["[UNK]"])
                    push!(result, token_id)
                else
                    push!(result, token_str)
                end
            end
        end
    end
    
    # Add [SEP] if needed
    if add_special_tokens
        if token_ids
            push!(result, tokenizer.special_tokens["[SEP]"])
        else
            push!(result, "[SEP]")
        end
    end
    
    # Cache the result if we're returning IDs
    if token_ids
        tokenizer.cache[cache_key] = copy(result)
    end
    
    return result
end

function TextEncodeBase.encode(tokenizer::ModernBertTokenizer, text::AbstractString)
    # Tokenize with special tokens
    token_ids = tokenize(tokenizer, text; token_ids=true, add_special_tokens=true)
    
    # Truncate to 512 tokens if needed
    if length(token_ids) > 512
        token_ids = token_ids[1:512]
        token_ids[end] = tokenizer.special_tokens["[SEP]"]  # Ensure we end with [SEP]
    end
    
    # Create attention mask and token type IDs
    token_type_ids = zeros(Int, length(token_ids))
    attention_mask = ones(Int, length(token_ids))
    
    return token_ids, token_type_ids, attention_mask
end

function TextEncodeBase.encode(tokenizer::ModernBertTokenizer, texts::Vector{String})
    # Process each text and return the tokens with their type IDs and attention masks
    results = [encode(tokenizer, text) for text in texts]
    token_ids = [r[1] for r in results]
    token_type_ids = [r[2] for r in results]
    attention_masks = [r[3] for r in results]
    return token_ids, token_type_ids, attention_masks
end

end # module ModernBertTokenizerImpl
