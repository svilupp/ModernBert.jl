module ModernBert

# Core dependencies
using JSON3
using Downloads
using Base: ones, zeros

# Import TextEncodeBase types and methods
using TextEncodeBase
import TextEncodeBase: AbstractTokenizer, encode, tokenize

# Include core tokenizer implementation
include("minimal_tokenizer.jl")

# Import and re-export from ModernBertTokenizerImpl
using .ModernBertTokenizerImpl: ModernBertTokenizer, tokenize, encode, load_modernbert_tokenizer, vocab_size
export ModernBertTokenizer, tokenize, encode, load_modernbert_tokenizer, vocab_size

# Include optional functionality
include("embedding.jl")
include("huggingface.jl")

# Re-export huggingface functionality
using .ModernBertHuggingFace: download_config_files
export BertModel, embed, download_config_files

end # module ModernBert
