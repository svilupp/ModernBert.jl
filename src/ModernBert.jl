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

export BPETokenizer, load_tokenizer
export tokenize, encode
include("bpe.jl")

# export encode
# include("encoder.jl")

export BertModel, embed
include("embedding.jl")

export download_config_files
include("huggingface.jl")

end # module
