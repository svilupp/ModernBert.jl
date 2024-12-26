# This file is kept for backward compatibility.
# All functionality has been moved to bytepair.jl

# Re-export types and functions
export ModernBertTokenizer, load_modernbert_tokenizer, tokenize, encode

# Deprecated type alias
const BPETokenizer = ModernBertTokenizer
Base.@deprecate_binding BPETokenizer ModernBertTokenizer false

# Deprecated function alias
@deprecate load_tokenizer load_modernbert_tokenizer
