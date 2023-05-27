using Pkg
Pkg.activate(".")

using LiveServer

include("make.jl")
serve(; dir = "docs/build/")
