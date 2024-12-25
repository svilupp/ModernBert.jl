module ModernBert

using DataDeps
using DoubleArrayTries
using OrderedCollections
const DAT = DoubleArrayTries
using Downloads
using JSON3
using ONNXRunTime
import ONNXRunTime as ORT  # Use high-level API
using Statistics
using StringViews
using Unicode
using WordTokenizers

export BPETokenizer, create_bpe_tokenizer, load_tokenizer
export tokenize, bpe_encode, get_pairs, get_token_id
include("bpe.jl")

export encode
include("encoder.jl")

export BertModel, embed, mean_pooling
include("embedding.jl")

export download_config_files
include("huggingface.jl")

end # module
