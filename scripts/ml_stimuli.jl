"""
This script trains a neural network
whose weights are a measurement matrix.
It then uses those weights as stimuli
for a simulated tinnitus reconstruction experiment
using the synthetic subject.
"""

push!(LOAD_PATH, "..")

using Flux
using TinnitusReconstructor
using LinearAlgebra
using StatsBase: sample

# %% Constants

const n_trials = 200
const n_bins = 32

# %% Hyperparameters

# learning rate
const η = 0.001 # [1e-6, 5e-3]
# ADAM momementum
const β = (0.9, 0.999) # [0.9, 1] (K&U used ADAM not ADAMW)
# weight decay
const decay = 0
# Gaussian kernel standard deviation
const σs = [2, 5, 10, 20, 40, 80]
# Batch size
const B = 16 # [150, 1500]

# %% Data parameters

const n_training = 50_000
const n_val = 10_000

# sparsity
const p = 5

# %% Useful functions

@doc """
    loss(x, x̂; λ=0)

Compute the loss function,
which is mean maximum discrepancy
plus an L1 loss.
"""
function loss(x, x̂; λ=0, σs=[1])
    mmd_loss = sum(TinnitusReconstructor.mmd(x, x̂, σ) for σ in σs)
    l1_loss = λ * norm(x̂, 1)
    return mmd_loss + l1_loss
end

"""
    generate_data(n_samples::T, n::T, p::T) where T <: Integer

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

# %% Create the "neural network"

model = Dense(n_bins => n_trials, identity)
