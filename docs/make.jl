using OCR
using Documenter

DocMeta.setdocmeta!(OCR, :DocTestSetup, :(using OCR); recursive=true)

makedocs(;
    modules=[OCR],
    authors="Lilith Hafner <Lilith.Hafner@gmail.com> and contributors",
    repo="https://github.com/LilithHafner/OCR.jl/blob/{commit}{path}#{line}",
    sitename="OCR.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://LilithHafner.github.io/OCR.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/LilithHafner/OCR.jl",
    devbranch="main",
)
