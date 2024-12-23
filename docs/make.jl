using ModernBert
using Documenter

DocMeta.setdocmeta!(ModernBert, :DocTestSetup, :(using ModernBert); recursive=true)

makedocs(;
    modules=[ModernBert],
    authors="J S <49557684+svilupp@users.noreply.github.com> and contributors",
    sitename="ModernBert.jl",
    format=Documenter.HTML(;
        canonical="https://svilupp.github.io/ModernBert.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/svilupp/ModernBert.jl",
    devbranch="main",
)
