using BytePairEncoding
using TextEncodeBase
using JSON3
using ModernBert
using BytePairEncoding: BPE, BPETokenization, BPETokenizer, GPT2Tokenization, Merge, parse_merge, MString
using TextEncodeBase: encode, tokenize, FlatTokenizer, CodeNormalizer
using TextEncodeBase: Sentence, TokenStages, TokenStage, SentenceStage, WordStage, ParentStages, getvalue

# Load test configuration
config_path = joinpath(@__DIR__, "..", "test", "model", "tokenizer.json")
config = JSON3.read(read(config_path))

# Initialize tokenizer components separately for analysis
println("Initializing tokenizer components...")

# Create basic BPE merges
bpe_merges = Dict{Tuple{Merge, Merge}, Int}()
for (i, merge_entry) in enumerate(config.model.merges)
    merge_str = string(merge_entry)
    merge_parts = split(merge_str)
    if length(merge_parts) == 2
        merge_pair = (parse_merge(merge_parts[1]), parse_merge(merge_parts[2]))
        bpe_merges[merge_pair] = i
    end
end

println("Number of merge rules: ", length(bpe_merges))
println("\nFirst 5 merge rules:")
for (pair, id) in Iterators.take(bpe_merges, 5)
    println("$pair => $id")
end

# Initialize tokenizer stages
base_tokenizer = BPE(bpe_merges)
gpt2_tok = GPT2Tokenization()
bpe_tok = BPETokenization(gpt2_tok, base_tokenizer)
code_norm = CodeNormalizer(bpe_tok, gpt2_codemap())
tokenizer = FlatTokenizer(code_norm)

# Test sentences from failing tests
test_sentences = [
    "The capital of France is [MASK].",
    "Python is a great programming language."
]

println("\nAnalyzing tokenization process...")
for sentence in test_sentences
    println("\nProcessing: ", sentence)
    
    # Step 1: GPT2 Tokenization
    println("\nStep 1: GPT2 Tokenization")
    gpt2_result = gpt2_tok(Sentence(sentence))
    println("GPT2 tokens: ", join([getvalue(t) for t in gpt2_result], " | "))
    
    # Step 2: BPE Application
    println("\nStep 2: BPE Tokenization")
    bpe_result = bpe_tok(Sentence(sentence))
    println("BPE tokens: ", join([getvalue(t) for t in bpe_result], " | "))
    
    # Step 3: Code Normalization
    println("\nStep 3: Code Normalization")
    norm_result = code_norm(Sentence(sentence))
    println("Normalized tokens: ", join([getvalue(t) for t in norm_result], " | "))
    
    # Step 4: Final Flat Tokenization
    println("\nStep 4: Final Tokenization")
    final_result = tokenizer(Sentence(sentence))
    println("Final tokens: ", join([getvalue(t) for t in final_result], " | "))
end

# Compare with expected token IDs
println("\nComparing with expected token IDs...")
expected_ids = Dict(
    "The capital of France is [MASK]." => [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282],
    "Python is a great programming language." => [50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282]
)

# Initialize ModernBert tokenizer
modernbert_tokenizer = ModernBertTokenizer(config_path)

for (sentence, expected) in expected_ids
    tokens, _, _ = encode(modernbert_tokenizer, sentence)
    println("\nSentence: ", sentence)
    println("Expected: ", expected)
    println("Got:      ", tokens)
    
    # Print token values for debugging
    println("\nToken mapping details:")
    for (exp, got) in zip(expected, tokens)
        exp_token = get(modernbert_tokenizer.id_to_token, exp, "UNKNOWN")
        got_token = get(modernbert_tokenizer.id_to_token, got, "UNKNOWN")
        println("Expected $exp ($exp_token) -> Got $got ($got_token)")
    end
end
