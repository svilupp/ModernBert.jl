using Test
using ModernBert
using JSON3

@testset "BPE Tokenizer Integration Tests" begin
    model_dir = joinpath(@__DIR__, "model")
    vocab_path = joinpath(model_dir, "tokenizer.json")
    @test isfile(vocab_path) "tokenizer.json not found"
    
    # Create tokenizer directly from config
    bpe = create_bpe_tokenizer(vocab_path)
    
    @testset "Vocabulary Loading" begin
        # Test vocabulary size
        @test length(bpe.vocab) > 0 "Vocabulary should not be empty"
        
        # Test special tokens
        special_tokens = ["[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"]
        for token in special_tokens
            @test haskey(bpe.special_tokens, token) "Special token $token missing"
        end
        
        # Test special token IDs
        @test bpe.special_tokens["[CLS]"] == 50281
        @test bpe.special_tokens["[SEP]"] == 50282
        @test bpe.special_tokens["[MASK]"] == 50284
        @test bpe.special_tokens["[PAD]"] == 50283
        @test bpe.special_tokens["[UNK]"] == 50280
    end
    
    @testset "BPE Encoding" begin
        # Test basic BPE encoding
        tokens = bpe("Hello world!"; token_ids=false)
        @test length(tokens) > 0 "Should produce tokens"
        @test !any(startswith.(tokens, "##")) "BPE should not use WordPiece '##' prefix"
        
        # Test token ID conversion
        token_ids = bpe("Hello world!"; token_ids=true)
        @test all(id -> id isa Integer, token_ids) "Token IDs should be integers"
        @test length(token_ids) == length(tokens) "Number of tokens should match number of IDs"
    end
    
    @testset "Edge Cases" begin
        # Empty string
        empty_tokens = bpe(""; token_ids=false)
        @test isempty(empty_tokens) "Empty string should produce no tokens"
        
        # Whitespace
        ws_tokens = bpe("   "; token_ids=false)
        @test !isempty(ws_tokens) "Whitespace should be tokenized"
        
        # Special characters
        special_chars = "!@#$%^&*()"
        special_tokens = bpe(special_chars; token_ids=false)
        @test !isempty(special_tokens) "Special characters should be tokenized"
        
        # Unicode
        unicode_text = "Hello 世界"
        unicode_tokens = bpe(unicode_text; token_ids=false)
        @test !isempty(unicode_tokens) "Unicode should be tokenized"
    end
    
    @testset "Token ID Consistency" begin
        text = "The quick brown fox jumps over the lazy dog"
        
        # Get tokens and IDs separately
        tokens = bpe(text; token_ids=false)
        token_ids = bpe(text; token_ids=true)
        
        # Manual ID lookup
        manual_ids = Int[]
        for token in tokens
            id = get(bpe.special_tokens, token, nothing)
            if isnothing(id)
                id = get(bpe.vocab, token, bpe.special_tokens["[UNK]"])
            end
            push!(manual_ids, id)
        end
        
        @test token_ids == manual_ids "Token ID conversion should be consistent"
    end
end
