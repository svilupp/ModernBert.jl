using Test
using ModernBert
using JSON3

@testset "Special Token IDs" begin
    # Load tokenizer with vocabulary file
    vocab_path = joinpath(@__DIR__, "model", "tokenizer.json")
    @assert isfile(vocab_path) "tokenizer.json not found at $(vocab_path)"
    tokenizer = load_modernbert_tokenizer(vocab_path)
    
    # Test special token IDs match exactly
    @test tokenizer.special_tokens["[UNK]"] == 50280
    @test tokenizer.special_tokens["[CLS]"] == 50281
    @test tokenizer.special_tokens["[SEP]"] == 50282
    @test tokenizer.special_tokens["[PAD]"] == 50283
    @test tokenizer.special_tokens["[MASK]"] == 50284
    
    # Test basic tokenization with actual token IDs
    text = "The capital of France is [MASK]."
    tokens, token_types, attention_mask = encode(tokenizer, text)
    @test tokens[1] == 50281  # [CLS]
    @test tokens[2] == 510    # The
    @test tokens[3] == 38479  # capital
    @test tokens[4] == 1171   # of
    @test tokens[5] == 33639  # France
    @test tokens[6] == 261    # is
    @test tokens[7] == 50284  # [MASK]
    @test tokens[8] == 15     # .
    @test tokens[9] == 50282  # [SEP]
    
    # Test [UNK] token with unknown word
    text = "unknown_token_xyz"
    tokens, token_types, attention_mask = encode(tokenizer, text)
    @test tokens[1] == 50281  # [CLS]
    @test 50280 in tokens     # [UNK]
    @test tokens[end] == 50282  # [SEP]
end
