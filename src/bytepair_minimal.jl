module ModernBertTokenizerImpl

# Export types and functions first
export ModernBertTokenizer, encode, tokenize, load_modernbert_tokenizer, add_special_tokens

# Import dependencies
using TextEncodeBase
using BytePairEncoding: BPE, BPETokenization, BPETokenizer, GPT2Tokenization, Merge, parse_merge, gpt2_codemap
using TextEncodeBase: encode, tokenize, FlatTokenizer, CodeNormalizer
using TextEncodeBase: Sentence, TokenStages, TokenStage, SentenceStage, WordStage, ParentStages, getvalue
using JSON3
using Base: ones, zeros

# Import functions we want to extend
import TextEncodeBase: tokenize, encode

# Define module-level constants and flags
const include_special_tokens = true  # Default flag for special token inclusion
const token_ids = true  # Default flag for returning token IDs instead of strings

# Define module-level constants
const DEFAULT_SPECIAL_TOKENS = Dict{String, Int}(
    "[UNK]" => 50280,
    "[CLS]" => 50281,
    "[SEP]" => 50282,
    "[PAD]" => 50283,
    "[MASK]" => 50284,
    " " => 50273,      # Space token
    "  " => 50274,     # Double space token
    "   " => 50275,    # Triple space token
    "\n" => 50286,     # Newline token
    "\t" => 50287      # Tab token
)

"""
    ModernBertTokenizer

A tokenizer implementation that combines BytePairEncoding with special token handling.
"""
mutable struct ModernBertTokenizer
    tokenizer::Any  # BytePairEncoding tokenizer
    vocab::Dict{String, Int}
    special_tokens::Dict{String, Int}
    id_to_token::Dict{Int, String}
    cache::Dict{String, Vector{Int}}
end

"""
    add_special_tokens(tokenizer::ModernBertTokenizer)

Add special tokens to the tokenizer's vocabulary and special_tokens dictionaries.
"""
function add_special_tokens(tokenizer::ModernBertTokenizer)
    # Update tokenizer's special_tokens dictionary with module-level constants
    tokenizer.special_tokens = DEFAULT_SPECIAL_TOKENS
    
    # Add special tokens to vocabulary if not present
    for (token, id) in DEFAULT_SPECIAL_TOKENS
        tokenizer.vocab[token] = id
        tokenizer.id_to_token[id] = token
    end
    
    return tokenizer
end

"""
    load_modernbert_tokenizer(config_path::String="test/model/tokenizer.json")

Convenience function to create a ModernBertTokenizer instance.
"""
function load_modernbert_tokenizer(config_path::String="test/model/tokenizer.json")
    ModernBertTokenizer(config_path)
end

"""
    ModernBertTokenizer(config_path::String)

Create a ModernBertTokenizer from a configuration file.
"""
function ModernBertTokenizer(config_path::String)
    config = JSON3.read(read(config_path, String))
    
    # Use module-level special tokens mapping
    special_tokens = copy(DEFAULT_SPECIAL_TOKENS)
    
    # Create vocabulary mapping directly from config
    vocab = Dict{String, Int}()
    
    # Load vocabulary with exact GPT2 token mappings
    for (token, id) in config.model.vocab
        token_str = String(token)
        token_id = parse(Int, string(id))
        # Store token exactly as in GPT2 vocabulary
        vocab[token_str] = token_id
        
        # For prefixed tokens, also store non-prefixed version as fallback
        if startswith(token_str, 'Ġ')
            base_token = token_str[nextind(token_str, 1):end]
            if !haskey(vocab, base_token)
                vocab[base_token] = token_id
            end
        end
    end
    
    # Then add special tokens, overwriting any conflicts
    for (token, id) in special_tokens
        vocab[token] = id
    end
    
    # Initialize GPT2 tokenizer with proper configuration
    bpe_merges = Dict{Tuple{Merge, Merge}, Int}()
    for (i, merge_entry) in enumerate(config.model.merges)
        merge_str = string(merge_entry)
        # Use BytePairEncoding's built-in parser
        try
            pair = parse_merge(merge_str)
            bpe_merges[pair] = i
        catch e
            # Skip invalid merge rules
            @warn "Skipping invalid merge rule: $merge_str"
        end
    end
    
    # Create tokenizer following BytePairEncoding.jl's test_bbpe.jl example
    # Initialize base BPE with merges
    base_tokenizer = BPE(bpe_merges)
    
    # Create tokenizer pipeline exactly as in test_bbpe.jl
    tokenizer = FlatTokenizer(
        CodeNormalizer(
            BPETokenization(
                GPT2Tokenization(),  # Use default GPT2Tokenization
                base_tokenizer       # BPE with loaded merges
            ),
            gpt2_codemap()          # Use default GPT2 codemap
        )
    )
    
    # Create reverse mapping for id_to_token
    id_to_token = Dict{Int, String}()
    for (token, id) in vocab
        id_to_token[id] = token
    end
    for (token, id) in special_tokens
        id_to_token[id] = token
    end
    
    ModernBertTokenizer(tokenizer, vocab, special_tokens, id_to_token, Dict{String, Vector{Int}}())
end

function tokenize(tokenizer::ModernBertTokenizer, text::String; token_ids::Bool=true, include_special_tokens::Bool=true)
    # Initialize result array
    result = token_ids ? Int[] : String[]
    
    # Add CLS token if requested
    if include_special_tokens
        if token_ids
            push!(result, tokenizer.special_tokens["[CLS]"])
        else
            push!(result, "[CLS]")
        end
    end
    
    # Handle special case when text is exactly a special token
    if haskey(tokenizer.special_tokens, text)
        if token_ids
            push!(result, tokenizer.special_tokens[text])
        else
            push!(result, text)
        end
    else
        # Pre-process text to handle special tokens
        parts = String[]
        current_text = text
        
        # Process special tokens directly without creating temporary dictionary
        # This avoids unnecessary allocations and dictionary operations
        for token in ("[MASK]", "[SEP]", "[CLS]", "[PAD]", "[UNK]")
            current_text = replace(current_text, token => token)
        end
        
        # Split text into parts while preserving special tokens
        parts = String[]
        current_pos = 1
        text_length = length(current_text)
        
        while current_pos <= text_length
            # Find the next special token more efficiently
            next_special = nothing
            next_pos = text_length + 1
            
            # Direct iteration over special tokens without dictionary lookup
            for token in ("[MASK]", "[SEP]", "[CLS]", "[PAD]", "[UNK]")
                pos = findnext(token, current_text, current_pos)
                if !isnothing(pos) && pos.start < next_pos
                    next_special = (token, token, pos)
                    next_pos = pos.start
                end
            end
            
            if !isnothing(next_special)
                token, placeholder, pos = next_special
                # Add text before special token if any
                if pos.start > current_pos
                    push!(parts, current_text[current_pos:prevind(current_text, pos.start)])
                end
                # Add the original special token
                push!(parts, token)
                current_pos = pos.start + length(placeholder)
            else
                # Add remaining text
                push!(parts, current_text[current_pos:end])
                break
            end
        end
        
        # If no parts were added, use the entire text
        if isempty(parts)
            push!(parts, current_text)
        end
        
        # Process each part
        for (i, part) in enumerate(parts)
            # Check if part is a special token
            original_token = nothing
            if haskey(tokenizer.special_tokens, part)
                original_token = part
            end
            
            if !isnothing(original_token)
                # Handle special tokens
                if token_ids
                    push!(result, tokenizer.special_tokens[original_token])
                else
                    push!(result, original_token)
                end
            else
                # Process regular text using BytePairEncoding's native pipeline
                # Handle punctuation more efficiently
                processed_parts = String[]
                current_part = ""
                
                # Single-pass punctuation handling
                for (i, char) in enumerate(part)
                    if char in ('.', ',', '!', '?')
                        if !isempty(current_part)
                            push!(processed_parts, current_part)
                            current_part = ""
                        end
                        push!(processed_parts, string(char))
                    else
                        current_part *= string(char)
                    end
                end
                
                # Add any remaining text
                if !isempty(current_part)
                    push!(processed_parts, current_part)
                end
                
                # Process each subpart
                for (k, subpart) in enumerate(processed_parts)
                    # Skip empty parts
                    isempty(strip(subpart)) && continue
                    
                    # Normalize whitespace
                    normalized_part = strip(subpart)
                    is_first_in_text = i == 1 && k == 1 && all(isspace, subpart[1:prevind(subpart, firstindex(normalized_part))])
                    
                    # Process tokens
                    if !isempty(normalized_part)
                        tokens = tokenizer.tokenizer(Sentence(normalized_part))
                        
                        for (j, token) in enumerate(tokens)
                            token_str = getvalue(token)
                            isempty(token_str) && continue
                            
                            if token_ids
                                # Handle punctuation tokens directly
                                if length(token_str) == 1 && occursin(punctuation, token_str)
                                    token_id = get(tokenizer.vocab, token_str, nothing)
                                    if isnothing(token_id)
                                        token_id = tokenizer.special_tokens["[UNK]"]
                                    end
                                else
                                    # Optimize token lookup with single dictionary access
                                    needs_prefix = (!is_first_in_text && j == 1 && k == 1) || 
                                                 (j > 1) || (k > 1 && !occursin(punctuation, token_str))
                                    
                                    # Try lookup with existing string first
                                    token_id = get(tokenizer.vocab, token_str, nothing)
                                    
                                    # Only modify string if needed
                                    if isnothing(token_id)
                                        if needs_prefix && !startswith(token_str, 'Ġ')
                                            # Try with prefix
                                            token_id = get(tokenizer.vocab, "Ġ" * token_str, nothing)
                                        elseif startswith(token_str, 'Ġ')
                                            # Try without prefix
                                            token_id = get(tokenizer.vocab, @view(token_str[nextind(token_str, 1):end]), nothing)
                                        end
                                    end
                                end
                                
                                # If still not found, use UNK token
                                if isnothing(token_id)
                                    token_id = tokenizer.special_tokens["[UNK]"]
                                end
                                
                                push!(result, token_id)
                            else
                                push!(result, token_str)
                            end
                        end
                    end
                end
            end
        end
    end
    
    # Add SEP token if requested
    if include_special_tokens
        if token_ids
            push!(result, tokenizer.special_tokens["[SEP]"])
        else
            push!(result, "[SEP]")
        end
    end
    
    return result
end

function encode(tokenizer::ModernBertTokenizer, text::String)
    # Special case: if text is a special token, wrap it with CLS/SEP
    if haskey(tokenizer.special_tokens, text)
        tokens = [
            tokenizer.special_tokens["[CLS]"],
            tokenizer.special_tokens[text],
            tokenizer.special_tokens["[SEP]"]
        ]
    else
        # For regular text, use tokenize with special tokens
        tokens = tokenize(tokenizer, text; token_ids=true, include_special_tokens=true)
    end
    attention_mask = ones(Int, length(tokens))
    token_type_ids = zeros(Int, length(tokens))
    return tokens, attention_mask, token_type_ids
end

function encode(tokenizer::ModernBertTokenizer, texts::Vector{String})
    # Get individual encodings
    encodings = [encode(tokenizer, text) for text in texts]
    
    # Find max length
    max_length = maximum(length(enc[1]) for enc in encodings)
    
    # Create matrices
    n_texts = length(texts)
    token_matrix = fill(tokenizer.special_tokens["[PAD]"], max_length, n_texts)
    attention_matrix = zeros(Int, max_length, n_texts)
    type_matrix = zeros(Int, max_length, n_texts)
    
    # Fill matrices
    for (j, encoding) in enumerate(encodings)
        tokens, attention, types = encoding
        for i in 1:length(tokens)
            token_matrix[i,j] = tokens[i]
            attention_matrix[i,j] = attention[i]
            type_matrix[i,j] = types[i]
        end
    end
    
    return token_matrix, attention_matrix, type_matrix
end

# Constructor function for backward compatibility
function ModernBertTokenizer()
    config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
    ModernBertTokenizer(config_path)
end

end  # ModernBertTokenizerImpl
