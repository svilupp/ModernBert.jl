module AnalyzeGPT2
using BytePairEncoding
using TextEncodeBase
using BytePairEncoding: GPT2Tokenization, gpt2_codemap

function analyze_gpt2_tokens()
    # Create base GPT2 tokenization
    base_tkr = GPT2Tokenization()
    
    # Analyze specific bytes in codemap
    println("\nGPT2 Codemap analysis:")
    codemap = gpt2_codemap()
    test_bytes = [0x20, 0x0A, 0x09, 0x0D, 0x21, 0x22, 0x23]  # space, \n, \t, \r, !, ", #
    for byte in test_bytes
        if haskey(codemap, byte)
            println("Byte $(repr(Char(byte))) ($(byte)) => $(codemap[byte])")
        end
    end
    
    # Test special characters and byte-level encoding
    test_strings = [
        "Hello",
        " Hello",  # with leading space
        "world",
        " world",  # with leading space
        "Ġworld",  # with GPT-2 space token
        "Hello world",
        "Hello  world",  # double space
        "Hello\nworld",  # newline
        "Hello\tworld",  # tab
        "\u0120",  # GPT-2 space token (Ġ)
        "\u0120Hello",  # GPT-2 space + word
        "\n",  # newline
        "\t",  # tab
        " ",   # space
        "  ",  # double space
        "\u0120\u0120",  # double GPT-2 space
        "!\"#\$%&'()*+,-./:;<=>?@[\\]^_`{|}~",  # all special characters
    ]
    
    println("\nTokenization examples:")
    for str in test_strings
        sentence = Sentence(str)
        result = base_tkr(sentence)
        println("\nInput: '", str, "'")
        println("Raw bytes: ", [Int(c) for c in str])
        println("Tokens: ", [String(getvalue(t)) for t in result])
    end
    
    # Check for special byte tokens
    println("\nChecking special byte tokens:")
    special_bytes = [0x20, 0x0A, 0x09, 0x0D]  # space, newline, tab, carriage return
    for byte in special_bytes
        char = Char(byte)
        sentence = Sentence(string(char))
        result = base_tkr(sentence)
        println("\nByte $byte ($(repr(char))):")
        println("Tokens: ", [String(getvalue(t)) for t in result])
    end
end
end # module

using .AnalyzeGPT2
AnalyzeGPT2.analyze_gpt2_tokens()
