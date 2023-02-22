"""
This script trains a neural network
whose weights are a measurement matrix.
It then uses those weights as stimuli
for a simulated tinnitus reconstruction experiment
using the synthetic subject.
"""

push!(LOAD_PATH, "..")

using Optimisers
using Zygote
using TinnitusReconstructor
using LinearAlgebra
using StatsBase: sample
using MLUtils
using ProgressMeter
using FileIO

# %% Constants

const n_trials = 100
const n_bins = 16

# %% Hyperparameters

# learning rate
const η = 0.001f0 # [1e-6, 5e-3]
# ADAM momementum
const β = (0.9f0, 0.999f0) # [0.9, 1] (K&U used ADAM not ADAMW)
# weight decay
const decay = 0.0f0
# Gaussian kernel standard deviation
const σs = [2, 5, 10, 20, 40, 80]
# Batch size
const B = 4 # [150, 1500]
# L1 loss coefficient
const λ = 0.001f0

# %% Data parameters

const n_training = 50_000
const n_val = 10_000

# sparsity
const p = 3

# %% Useful functions

# TODO: fix the regularization
@doc """
    mmd_loss(x, x̂; σs=[1])

Compute the mean maximum discrepancy loss
with a Gaussian kernel.
`σs` is a list of kernel sizes (standard deviations)
that the loss is summed over.

# Examples
```jldoctest
julia> mmd_loss(1, 1; σs=[1, 2, 3])
0.0

julia> mmd_loss(1, 1)
0.0

julia> mmd_loss(1, 2)
0.7869386805747332
```

# See Also

* [TinnitusReconstructor.mmd](@ref TinnitusReconstructor.mmd)
"""
function mmd_loss(x, x̂; σs=[1])
    return sum(TinnitusReconstructor.mmd(x, x̂, σ) for σ in σs)
end

"""
    generate_data(n_samples::T, m::T, n::T, p::T) where {T<:Integer}

Generate `n_samples` training samples of length `n`
with sparsity `p`.
The size of `H` is `n × n_samples`
and the size of `U` is `m x n_samples`.
Note that these samples are sparse in the standard basis (identity matrix).
"""
function generate_data(n_samples::T, m::T, n::T, p::T) where {T<:Integer}
    # Generate a Gaussian random matrix
    H = randn(Float32, n, n_samples) ./ p
    # Set all but p indices in each row to zero
    for h in eachcol(H)
        indices = sample(1:n, n - p; replace=false)
        h[indices] .= 0
    end
    # Rescale
    H /= sqrt(norm(H) / n_samples)

    # Compute the label data
    U = randn(Float32, m, n_samples)
    for u in eachcol(U)
        u .= u / norm(u)
    end
    return H, U
end

@doc """
    model(x, W)

Parameterize the optimization problem as a model
with input `x` and weights `W`.
"""
function model(x, W)
    W̄ = abs.(W)
    return W̄ * x
end

function main()
    # %% Create the training data
    H, U = generate_data(32, n_trials, n_bins, p)
    dataloader = MLUtils.DataLoader((H, U); batchsize=B)

    # Instantiate parameters
    W = rand(Float32, n_trials, n_bins)
    state = Optimisers.setup(Optimisers.Adam(η, β), W)

    ProgressMeter.@showprogress for (h, u) in dataloader
        # Zygote.gradient(W -> loss(model(h, W), u), W)
        L, Δ = Zygote.withgradient(W) do W
            this_mmd_loss = mmd_loss(model(h, W), u; σs=σs)
            this_l1_loss = λ * norm(W, 1)
            this_mmd_loss + this_l1_loss
        end

        Optimisers.update(state, W, Δ[1])
        @show round(L, digits=3)
    end

    return save("weights.jld2", abs.(W))
end

main()