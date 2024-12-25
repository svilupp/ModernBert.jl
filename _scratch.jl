using ModernBert, JSON3

MODEL_PATH = joinpath(@__DIR__, "data", "model.onnx")
model = BertModel(model_path = MODEL_PATH)
encoder = model.encoder

vocab_path = joinpath(@__DIR__, "data", "tokenizer.json")
config = JSON3.read(read(vocab_path));
inv = Dict(v => k for (k, v) in pairs(config.model.vocab))

inv[50280]

inv[328], inv[21002], inv[17254]
inv[891], inv[78], inv[181]

text = "The antidisestablishmentarianistically-minded pseudopneumonoultramicroscopicsilicovolcanoconiosis researcher hypothesized that supercalifragilisticexpialidocious manifestations of hippopotomonstrosesquippedaliophobia could be counterrevolutionarily interconnected with floccinaucinihilipilification tendencies among immunoelectrophoretically-sensitive microspectrophotofluorometrically-analyzed organophosphates..."
tokens1 = tokenize(encoder, text; token_ids = true)

whitespace_text = "   "
ws_tokens = tokenize(encoder, whitespace_text; token_ids = true)
ws_tokens, _, _ = encode(encoder, whitespace_text)
@assert ws_tokens == [50281, 50275, 50282]

text1 = "Hello world! This is a test."
tokens1 = tokenize(encoder, text1; token_ids = true, add_special_tokens = true)
@assert tokens1==[50281, 12092, 1533, 2, 831, 310, 247, 1071, 15, 50282] "Token IDs should match Python output"

# Test case 2: Masked sentence
text2 = "The capital of France is [MASK]."
tokens2 = tokenize(encoder, text2; token_ids = true, add_special_tokens = true)
@assert tokens2==[50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282] "Token IDs should match Python output for masked text"
tokens22, _, _ = encode(encoder, text2; add_special_tokens = true)
@assert tokens22 == [50281, 510, 5347, 273, 6181, 310, 50284, 15, 50282]

text3 = "The antidisestablishmentarianistically-minded researcher hypothesized about pseudoscientific phenomena."
tokens3, _, _ = encode(model, text3)
@test tokens3 == [50281, 510, 1331, 30861, 15425, 8922, 6656, 18260, 14, 23674,
    22780, 24045, 670, 10585, 5829, 850, 692, 16958, 15, 50282]