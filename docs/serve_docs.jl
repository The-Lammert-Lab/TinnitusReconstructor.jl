using Pkg
Pkg.activate(".")

using LiveServer

include("make.jl")
<<<<<<< HEAD
serve(dir="build/")
=======
serve(; dir="docs/build/")
>>>>>>> main
