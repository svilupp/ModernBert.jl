module ModernBertTokenizerImpl

using BytePairEncoding
using TextEncodeBase
using BytePairEncoding: BPE, BPETokenization, BPETokenizer, GPT2Tokenization, Merge, parse_merge
using TextEncodeBase: encode, tokenize, FlatTokenizer, CodeNormalizer
using TextEncodeBase: Sentence, TokenStages, TokenStage, SentenceStage, WordStage, ParentStages, getvalue
using BytePairEncoding: gpt2_codemap
using JSON3

# Import functions we want to extend
import TextEncodeBase: tokenize, encode

export ModernBertTokenizer, encode, tokenize, load_modernbert_tokenizer

# Convenience function to match test expectations
function load_modernbert_tokenizer(config_path::String="test/model/tokenizer.json")
    ModernBertTokenizer(config_path)
end

mutable struct ModernBertTokenizer
    tokenizer::Any  # BytePairEncoding tokenizer
    vocab::Dict{String, Int}
    special_tokens::Dict{String, Int}
    id_to_token::Dict{Int, String}
    cache::Dict{String, Vector{Int}}
end

"""
    ModernBertTokenizer(config_path::String)

Create a ModernBertTokenizer from a configuration file.
"""
function ModernBertTokenizer(config_path::String)
    config = JSON3.read(read(config_path, String))
    
    # Create special tokens mapping first
    special_tokens = Dict{String, Int}(
        "[UNK]" => 50280,  # Start with UNK token
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
    
    # Create vocabulary mapping, ensuring we don't overwrite special token IDs
    vocab = Dict{String, Int}()
    
    # First, add special tokens to vocabulary
    for (token, id) in special_tokens
        vocab[token] = id
    end
    
    # Then add regular vocabulary tokens, ensuring we don't use special token IDs
    for (token, id) in config.model.vocab
        token_str = String(token)
        if !haskey(vocab, token_str)  # Only add if not a special token
            vocab[token_str] = id
        end
    end
    
    # Use BytePairEncoding's GPT2 tokenizer directly
    tokenizer = BytePairEncoding.load_gpt2()
    
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

function tokenize(tokenizer::ModernBertTokenizer, text::String; token_ids::Bool=true, add_special_tokens::Bool=true)
    # Initialize result array
    result = token_ids ? Int[] : String[]
    
    # Handle special case when text is exactly a special token
    if haskey(tokenizer.special_tokens, text)
        # For special tokens, return only the token ID/text without any additional tokens
        return token_ids ? [tokenizer.special_tokens[text]] : [text]
    end
    
    # For regular text, add special tokens if requested
    if add_special_tokens
        if token_ids
            push!(result, tokenizer.special_tokens["[CLS]"])
        else
            push!(result, "[CLS]")
        end
    end

    # Split on special tokens while preserving them
    parts = String[]
    current_pos = 1
    last_pos = 1
    text_length = length(text)
    
    while current_pos <= text_length
        # Check for special tokens at current position
        found_special = false
        for token in ("[MASK]", "[SEP]", "[CLS]", "[PAD]", "[UNK]")
            token_length = length(token)
            # Calculate end position using nextind
            end_pos = current_pos
            for _ in 1:token_length
                if end_pos > text_length
                    break
                end
                end_pos = nextind(text, end_pos)
            end
            if end_pos <= text_length + 1  # +1 because end_pos points to next position
                current_substr = SubString(text, current_pos, prevind(text, end_pos))
                if current_substr == token
                    # Add text before special token if any
                    if current_pos > last_pos
                        prev_text = text[last_pos:prevind(text, current_pos)]
                        if !isempty(strip(prev_text))
                            # Handle whitespace before special token
                            if endswith(prev_text, " ")
                                base_text = rstrip(prev_text)
                                if !isempty(base_text)
                                    push!(parts, base_text)
                                end
                            else
                                push!(parts, prev_text)
                            end
                        end
                    end
                    # Add special token
                    push!(parts, token)
                    # Update positions
                    current_pos = nextind(text, current_pos + token_length - 1)
                    last_pos = current_pos
                    found_special = true
                    break
                end
            end
        end
        if !found_special
            current_pos = nextind(text, current_pos)
        end
    end
    
    # Add remaining text if any
    if last_pos <= text_length
        remaining_text = text[last_pos:end]
        if !isempty(strip(remaining_text))
            push!(parts, remaining_text)
        end
    end

    # Process each part
    for part in parts
        if haskey(tokenizer.special_tokens, part)
            if token_ids
                push!(result, tokenizer.special_tokens[part])
            else
                push!(result, part)
            end
        else
            # Skip empty parts
            isempty(strip(part)) && continue
            
            # Tokenize regular text using GPT2 tokenizer
            tokens = tokenizer.tokenizer(Sentence(part))
            for token in tokens
                token_str = String(token)
                # Skip empty tokens
                isempty(strip(token_str)) && continue
                
                if token_ids
                    # Try different token formats
                    token_id = nothing
                    
                    # Try different token formats
                    token_id = nothing
                    
                    # Handle pure space tokens first
                    if token_str == "Ġ" || token_str == " " || token_str == "\u0120"
                        token_id = tokenizer.special_tokens[" "]  # Space token ID (50273)
                    # Handle multiple consecutive spaces
                    elseif all(c -> c == ' ', token_str)
                        token_id = tokenizer.special_tokens[" "]  # Space token ID (50273)
                    else
                        # For all other tokens, try direct vocabulary lookup first
                        token_id = get(tokenizer.vocab, token_str, nothing)
                        
                        # If not found and starts with space/Ġ, try without prefix
                        if isnothing(token_id) && (startswith(token_str, 'Ġ') || startswith(token_str, "\u0120") || startswith(token_str, ' '))
                            rest = token_str[nextind(token_str, 1):end]
                            if !isempty(rest)
                                token_id = get(tokenizer.vocab, rest, nothing)
                            end
                        end
                    end
                    
                    # If still no token found, use UNK token
                    if isnothing(token_id)
                        token_id = tokenizer.special_tokens["[UNK]"]  # UNK token ID (50280)
                    end
                    
                    # If still no token found, use UNK token
                    if isnothing(token_id)
                        token_id = tokenizer.special_tokens["[UNK]"]  # UNK token ID (50280)
                    end
                    
                    # Only add non-empty tokens
                    if !isnothing(token_id)
                        push!(result, token_id)
                    else
                        push!(result, tokenizer.special_tokens["[UNK]"])
                    end
                else
                    if startswith(token_str, 'Ġ')
                        # Handle Ġ prefix using proper string indexing
                        rest_of_token = token_str[nextind(token_str, 1):end]
                        token_str = " " * rest_of_token
                    end
                    # Skip empty tokens
                    !isempty(strip(token_str)) && push!(result, token_str)
                end
            end
        end
    end
    
    # Add SEP token if requested and not processing a special token
    if add_special_tokens && !haskey(tokenizer.special_tokens, text)
        if token_ids
            push!(result, tokenizer.special_tokens["[SEP]"])
        else
            push!(result, "[SEP]")
        end
    end
    
    result
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
        # For regular text, use tokenize
        tokens = tokenize(tokenizer, text)
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

end # module
