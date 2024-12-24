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
using Downloads
using JSON3


include("wordpiece.jl")
include("tokenizer.jl")
include("encoder.jl")

export BertModel, encode, embed, download_config_files
include("embedding.jl")

export download_config_files
include("huggingface.jl")

function __init__()
end

end # module
