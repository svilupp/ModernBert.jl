using TextEncodeBase
using BytePairEncoding: BPEEncoder
using ModernBert: ModernBertTokenizer
using TextEncodeBase: Vocab, DictBackedLookupDict, PerforatedOverwritableLookupVector,
                      DATLookupVector

"""
    load_modernbert_encoder(config_path::String)

Load ModernBertTokenizer as a BPEEncoder from BytePairEncoding.jl

```julia-repl
julia> enc = load_modernbert_encoder("data/tokenizer.json")
BPEEncoder(...)

julia> enc("hello world")
3-element Vector{Int64}:
 15339
  1917
   264
```
"""
function load_modernbert_encoder(config_path::String)
    tokenizer = ModernBertTokenizer(config_path)
    # Create lookup vector with special tokens
    # Create a vector with size of max vocab id, fill with empty strings
    max_id = maximum(values(tokenizer.vocab))
    vocab_lookup = fill("", 1 + max_id)
    # Fill in the actual tokens at their corresponding indices
    for (token, id) in tokenizer.vocab
        vocab_lookup[1 + id] = token
    end
    @assert length(vocab_lookup)==1 + max_id "Vocab lookup vector length mismatch (expected $max_id, got $(length(vocab_lookup)))"
    # @assert length(unique(vocab_lookup))==length(vocab_lookup) "Vocab lookup vector is not unique"
    # Find duplicates using unique with index function
    unique_mask = unique(i -> vocab_lookup[i], eachindex(vocab_lookup))
    all_indices = Set(eachindex(vocab_lookup))
    duplicate_indices = setdiff(all_indices, unique_mask)

    # Print duplicates if any exist
    if !isempty(duplicate_indices)
        println("Found duplicate tokens:")
        for idx in duplicate_indices
            token = vocab_lookup[idx]
            # Find all positions where this token appears
            positions = findall(==(token), vocab_lookup)
            println("Token '", token, "' appears at indices: ", positions .- 1)
        end
    end

    vector = PerforatedOverwritableLookupVector(
        DATLookupVector(vocab_lookup),
        DictBackedLookupDict(
            tokenizer.special_tokens, Dict(v => k for (k, v) in tokenizer.special_tokens)))
    # Create vocab without unknown token since it's byte-level BPE
    vocab = Vocab(vector, "", 0)
    # Create a BPEEncoder using the underlying tokenizer and vocab
    return BPEEncoder(tokenizer.tokenizer, vocab)
end

enc = load_modernbert_encoder("data/tokenizer.json")

enc.decode(enc.encode("hello world"))
enc.tokenizer("hello world")
enc.encode("hello world")

# Load tokenizer config
config = JSON3.read(read("data/tokenizer.json", String))

# Create vocabulary mapping
vocab = Dict{String, Int}()
for (token, id) in config.model.vocab
    token_str = String(token)
    token_id = parse(Int, string(id))
    vocab[token_str] = token_id
end

# Look up specific token IDs
for id in [0]
    token = findfirst(p -> p == id, pairs(vocab))
    if token !== nothing
        println("Token ID $id = '$(token)'")
    else
        println("Token ID $id not found in vocabulary")
    end
end
