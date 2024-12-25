config

inv = Dict(v=>k for (k,v) in pairs(config.model.vocab))

inv[2190]
inv[2984], inv[41121], inv[3838]
inv[891], inv[78], inv[181]


text="The antidisestablishmentarianistically-minded pseudopneumonoultramicroscopicsilicovolcanoconiosis researcher hypothesized that supercalifragilisticexpialidocious manifestations of hippopotomonstrosesquippedaliophobia could be counterrevolutionarily interconnected with floccinaucinihilipilification tendencies among immunoelectrophoretically-sensitive microspectrophotofluorometrically-analyzed organophosphates..."
tokens1 = tokenize(encoder, text; token_ids=true)