using Test
using TextEncodeBase
using JSON3

# Test minimal tokenizer module loading
@testset "Module Loading Tests" begin
    # Try to load the module
    include("../src/minimal_tokenizer.jl")
    
    # Verify module loaded successfully
    @test isdefined(Main, :ModernBertTokenizerImpl)
    
    # Test basic tokenizer creation
    vocab = Dict{String, Int}()
    special_tokens = Dict{String, Int}(
        "[UNK]" => 50280,
        "[CLS]" => 50281,
        "[SEP]" => 50282,
        "[PAD]" => 50283,
        "[MASK]" => 50284
    )
    
    # Create tokenizer instance
    tokenizer = ModernBertTokenizerImpl.ModernBertTokenizer(vocab, special_tokens)
    
    # Verify tokenizer type
    @test tokenizer isa ModernBertTokenizerImpl.ModernBertTokenizer
end
