module ModernBert

using DataDeps
using DoubleArrayTries
const DAT = DoubleArrayTries
using Downloads
using JSON3
using ONNXRunTime
import ONNXRunTime as ORT  # Use high-level API
using Statistics
using StringViews
using Unicode
using WordTokenizers

include("bpe.jl")
include("tokenizer.jl")
include("encoder.jl")
include("embedding.jl")
include("huggingface.jl")

export ModernBertModel, encode, embed
export BPETokenizer, create_bpe_tokenizer, BertTextEncoder, tokenize
export download_config_files
export HuggingFace

function __init__()
end

end # module
