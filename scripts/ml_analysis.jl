using BSON
using Optimisers
using DataFrames
using Plots
using LinearAlgebra

@doc """
    model(x, W)
+
Parameterize the optimization problem as a model
with input `x` and weights `W`.
"""
function model(x, W)
    return W * x
end

dir = pwd()

data_files = [str for str in readdir(dir, sort=true) if startswith(str, "model-")]

data = [BSON.load(data_file) for data_file in data_files]
λ = [d[:λ] for d in data]
loss = [d[:loss] for d in data]
r = [d[:acc] for d in data]
W = [d[:opt_state].state[1] for d in data]

df = DataFrame(λ=λ, loss=loss, r=r, W=W)

this_W = sort(filter(row -> row.λ == 1.0, df), :loss)[!, :W][1]

plot(normalize(reshape(this_W, 16, 100)[:, 1:10]))

histogram(reshape(this_W, 16, 100)[:, 1:10])