push!(LOAD_PATH, "..")
using Documenter
using TinnitusReconstructor

DocMeta.setdocmeta!(TinnitusReconstructor, :DocTestSetup, :(using TinnitusReconstructor);
                    recursive = true)

makedocs(;
         modules = [TinnitusReconstructor],
         authors = "Alec Hoyland <alec.hoyland@posteo.net> and Nelson Barnett <nbarnett@wpi.edu>",
         repo = "https://github.com/The-Lammert-Lab/TinnitusReconstructor.jl/blob/{commit}{path}#{line}",
         sitename = "TinnitusReconstructor.jl",
         format = Documenter.HTML(;
                                  prettyurls = get(ENV, "CI", "false") == "true",
                                  canonical = "https://the-lammert-lab.github.io/TinnitusReconstructor.jl",
                                  edit_link = "main",
                                  assets = String[]),
         pages = [
             "Home" => "index.md",
             "Stimulus Generation" => "stimgens.md",
             "Stimulus Generation Methods" => "stimgen_methods.md",
             "Utility Functions" => "funcs.md",
         ])

deploydocs(; repo = "github.com/the-lammert-lab/TinnitusReconstructor.jl",
           devbranch = "main")
