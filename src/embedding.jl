struct BertModel
    session::Any  # Use Any to accommodate ORT model type
    encoder::ModernBertEncoder
end

function Base.show(io::IO, model::BertModel)
    print(
        io, "BertModel(session=$(typeof(model.session)), encoder=$(typeof(model.encoder)))")
end

function BertModel(;
        model_path::String = joinpath(@__DIR__, "..", "data", "model.onnx"),
        config_dir::Union{String, Nothing} = nothing,
        repo_url::Union{String, Nothing} = nothing
)
    # Download config files if repo_url is provided
    if !isnothing(repo_url)
        config_dir = download_config_files(repo_url, tempdir())
    end

    # Use default config directory if none provided
    config_dir = something(config_dir, dirname(model_path))

    # Load tokenizer configuration
    vocab_path = joinpath(config_dir, "tokenizer.json")

    # Create ModernBertEncoder with the vocabulary and special tokens
    encoder = ModernBertEncoder(vocab_path)

    # Initialize ONNX session with high-level API
    session = ORT.load_inference(model_path)

    return BertModel(session, encoder)
end

function TextEncodeBase.encode(model::BertModel, text::AbstractString)
    return encode(model.encoder, text)
end

function TextEncodeBase.encode(model::BertModel, texts::AbstractVector{<:AbstractString})
    return encode(model.encoder, texts)
end

function mean_pooling(
        token_embeddings::AbstractArray, attention_mask::AbstractArray; verbose::Bool = false, kwargs...)
    # Debug information
    if verbose
        @info "Token embeddings shape before processing: $(size(token_embeddings))"
        @info "Attention mask shape before processing: $(size(attention_mask))"
    end

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
    if verbose
        @info "Hidden states shape: $(size(hidden_states))"
        @info "Attention mask shape after reshape: $(size(mask))"
    end

    # Apply attention mask and compute mean
    masked_embeddings = hidden_states .* mask
    summed = sum(masked_embeddings, dims = 1)
    counts = sum(mask, dims = 1)

    # Normalize and ensure correct output shape
    normalized = dropdims(summed ./ max.(counts, 1), dims = 1)
    if verbose
        @info "Pre-L2 normalized shape: $(size(normalized))"
    end

    # Apply L2 normalization
    if ndims(normalized) == 1
        # Single string case
        l2_norm = sqrt(sum(normalized .^ 2))
        normalized = normalized ./ max(l2_norm, 1e-12)
    else
        # Batch case
        l2_norms = sqrt.(sum(normalized .^ 2, dims = 2))
        normalized = normalized ./ max.(l2_norms, 1e-12)
    end
    if verbose
        @info "Final L2-normalized shape: $(size(normalized))"
    end

    return normalized .|> Float32
end

function embed(model::BertModel,
        texts::Union{AbstractString, AbstractVector{<:AbstractString}};
        verbose::Bool = false, kwargs...)
    token_ids = encode(model, texts)
    # Create attention mask by detecting padding tokens using encoder's special token mapping
    pad_id = model.encoder.special_tokens["[PAD]"]
    attention_mask = token_ids .!= pad_id

    # Convert to Int64 and ensure correct shape
    verbose && @info "Token IDs shape: $(size(token_ids))"
    verbose && @info "Attention mask shape: $(size(attention_mask))"

    inputs = Dict(
        "input_ids" => Int64.(token_ids),
        "attention_mask" => Int64.(attention_mask)
    )

    logits = model.session(inputs)["logits"] .|> Float32
    verbose && @info "Logits shape: $(size(logits))"

    sentence_embedding = mean_pooling(logits, attention_mask; verbose, kwargs...)
    verbose && @info "Sentence embedding shape: $(size(sentence_embedding))"

    return sentence_embedding
end

function (m::BertModel)(
        text::Union{AbstractString, AbstractVector{<:AbstractString}}; kwargs...)
    return embed(m, text; kwargs...)
end