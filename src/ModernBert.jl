module ModernBert

# Core dependencies
using JSON3
using ONNXRunTime
const ORT = ONNXRunTime
using Downloads
using Base: ones, zeros
using TextEncodeBase
using BytePairEncoding
using BytePairEncoding: BPE, BPETokenization, BPETokenizer, GPT2Tokenization, Merge,
                        parse_merge, gpt2_codemap
using TextEncodeBase: encode, tokenize, FlatTokenizer, CodeNormalizer
using TextEncodeBase: Sentence, TokenStages, TokenStage, SentenceStage, WordStage,
                      ParentStages, getvalue
using Base: ones, zeros
# Import TextEncodeBase types and methods for extension
import TextEncodeBase: AbstractTokenizer, encode, tokenize

export ModernBertTokenizer, tokenize, encode, add_special_tokens
include("bytepair.jl")

export BertModel, embed
include("embedding.jl")

export download_config_files
include("huggingface.jl")

end # module ModernBert
