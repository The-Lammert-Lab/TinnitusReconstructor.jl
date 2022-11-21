using TinnitusReconstructor
using Documenter

DocMeta.setdocmeta!(TinnitusReconstructor, :DocTestSetup, :(using TinnitusReconstructor); recursive=true)

makedocs(;
    modules=[TinnitusReconstructor],
    authors="Alec Hoyland <alec.hoyland@posteo.net> and contributors",
    repo="https://github.com/alec-hoyland/TinnitusReconstructor.jl/blob/{commit}{path}#{line}",
    sitename="TinnitusReconstructor.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://alec-hoyland.github.io/TinnitusReconstructor.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/alec-hoyland/TinnitusReconstructor.jl",
    devbranch="main",
)
