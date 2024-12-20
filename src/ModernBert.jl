module ModernBert

using DataDeps
using DoubleArrayTries
const DAT = DoubleArrayTries
using Downloads
using HuggingFace
using JSON3
using ONNXRunTime
import ONNXRunTime as ORT  # Use high-level API
using Statistics
using StringViews
using Unicode
using WordTokenizers

include("wordpiece.jl")
include("tokenizer.jl")
include("huggingface.jl")
include("encoder.jl")

export ModernBertModel, encode, embed, download_config_files

struct ModernBertModel
    session::Any  # Use Any to accommodate ORT model type
    encoder::BertTextEncoder
end

function ModernBertModel(;
    model_path::String = joinpath(@__DIR__, "..", "data", "model.onnx"),
    config_dir::Union{String, Nothing} = nothing,
    repo_url::Union{String, Nothing} = nothing
)
    # Download config files if repo_url is provided
    if !isnothing(repo_url)
        config_dir = download_config_files(repo_url, tempdir())
    end

    # Use default config directory if none provided
    config_dir = something(config_dir, joinpath(@__DIR__, "..", "data"))

    # Load tokenizer configuration
    vocab_path = joinpath(config_dir, "tokenizer.json")
    vocab_config = JSON3.read(read(vocab_path))
    vocab = Dict{String,Int}(String(k) => v for (k, v) in vocab_config["model"]["vocab"])

    # Extract special tokens from added_tokens and add them to vocabulary
    special_tokens = Dict{String,Int}()
    for token in vocab_config["added_tokens"]
        token_content = String(token["content"])
        token_id = token["id"]
        special_tokens[token_content] = token_id
        vocab[token_content] = token_id  # Add to main vocabulary
    end

    # Create WordPiece tokenizer with the correct UNK token
    wp = WordPiece(collect(keys(vocab)), "[UNK]";
                   max_char=200,
                   subword_prefix="##")

    # Create encoder with special tokens (using actual tokens, not IDs)
    encoder = BertTextEncoder(wp, vocab;
                            startsym="[CLS]",
                            endsym="[SEP]",
                            padsym="[PAD]")

    # Initialize ONNX session with high-level API
    session = ORT.load_inference(model_path)

    return ModernBertModel(session, encoder)
end

function encode(model::ModernBertModel, text::AbstractString)
    return encode(model.encoder, text)
end

function encode(model::ModernBertModel, texts::AbstractVector{<:AbstractString})
    return encode(model.encoder, texts)
end

function mean_pooling(token_embeddings::AbstractArray, attention_mask::AbstractArray)
    # Debug information
    @info "Token embeddings shape before processing: $(size(token_embeddings))"
    @info "Attention mask shape before processing: $(size(attention_mask))"

    # Get the hidden states (first 1024 dimensions)
    hidden_size = 1024
    if ndims(token_embeddings) == 3
        # For batch processing or single string with sequence length
        hidden_states = token_embeddings[:, :, 1:hidden_size]
        # Ensure attention mask matches hidden states dimensions
        if size(attention_mask, 2) > 1
            # Batch case: maintain batch dimensions
            mask = attention_mask
        else
            # Single string case: reshape for broadcasting
            mask = reshape(attention_mask, :, 1)
        end
    else
        # Reshape if needed for single string
        hidden_states = reshape(token_embeddings, :, hidden_size)
        mask = reshape(attention_mask, :, 1)
    end
    @info "Hidden states shape: $(size(hidden_states))"
    @info "Attention mask shape after reshape: $(size(mask))"

    # Apply attention mask and compute mean
    masked_embeddings = hidden_states .* mask
    summed = sum(masked_embeddings, dims=1)
    counts = sum(mask, dims=1)

    # Normalize and ensure correct output shape
    normalized = dropdims(summed ./ max.(counts, 1), dims=1)
    @info "Pre-L2 normalized shape: $(size(normalized))"

    # Apply L2 normalization
    if ndims(normalized) == 1
        # Single string case
        l2_norm = sqrt(sum(normalized .^ 2))
        normalized = normalized ./ max(l2_norm, 1e-12)
    else
        # Batch case
        l2_norms = sqrt.(sum(normalized .^ 2, dims=2))
        normalized = normalized ./ max.(l2_norms, 1e-12)
    end
    @info "Final L2-normalized shape: $(size(normalized))"

    return normalized
end

function embed(model::ModernBertModel, text::AbstractString)
    token_ids, token_type_ids, attention_mask = encode(model, text)

    inputs = Dict(
        "input_ids" => reshape(Int64.(token_ids), :, 1),
        "attention_mask" => reshape(Int64.(attention_mask), :, 1)
    )

    outputs = model.session(inputs)
    logits = first(values(outputs))

    sentence_embedding = mean_pooling(logits, attention_mask)

    return reshape(sentence_embedding, 1024)
end

function embed(model::ModernBertModel, texts::AbstractVector{<:AbstractString})
    token_ids, token_type_ids, attention_mask = encode(model, texts)

    inputs = Dict(
        "input_ids" => Int64.(token_ids),
        "attention_mask" => Int64.(attention_mask)
    )

    outputs = model.session(inputs)
    logits = first(values(outputs))

    sentence_embeddings = mean_pooling(logits, attention_mask)

    return sentence_embeddings
end

function __init__()
end

end # module
