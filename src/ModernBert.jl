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

include("tokenizer.jl")
include("encoder.jl")

export BertModel, encode, embed, download_config_files
export BPETokenizer, create_bpe_tokenizer, BertTextEncoder, tokenize, add_special_tokens, add_end_token
include("embedding.jl")

export download_config_files
include("huggingface.jl")

function __init__()
end

end # module
