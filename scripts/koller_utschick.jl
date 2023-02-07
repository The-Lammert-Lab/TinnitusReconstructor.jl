"""
This script is a test to implement
the maximum mean discrepancy optimization
problem for designing a compressive sensing matrix
with structural constraints
by Koller & Utschick 2022.
"""

push!(LOAD_PATH, "..")

using Flux
using Flux.Data: DataLoader
using TinnitusReconstructor
using Random: AbstractRNG
using LinearAlgebra
using StatsBase: sample

## Dimensionality
const m = 10
const n = 2

## Hyperparameters

# learning rate
const η = 0.001 # [1e-6, 5e-3]
# ADAM momementum
const β = (0.9, 0.999) # [0.9, 1] (K&U used ADAM not ADAMW)
# weight decay
const decay = 0
# Gaussian kernel standard deviation
const σ = 2 # [2, 5, 10, 20, 40, 80]
# Batch size
const B = 128 # [150, 1500]

## Data parameters

# sparsity
p = 5 # TODO

## Useful functions

"""
    loss(x, x̂)

Compute the loss function for this model.
"""
loss(x, x̂) = TinnitusReconstructor.mmd(x, x̂, σ)

"""
    generate_data(n_training::T, n::T, p::T) where T <: Integer

Generate `n_training` training samples of length `n`
with sparsity `p`.
Note that these samples are sparse in the standard basis (identity matrix).
"""
function generate_data(n_training::T, m::T, n::T, p::T) where T <: Integer
    # Generate a Gaussian random matrix
    H = randn(Float32, n_training, n) ./ p
    # Set all but p indices in each row to zero
    for h in eachrow(H)
        indices = sample(1:n, n-p, replace=false)
        h[indices] .= 0
    end
    # Rescale
    H /= sqrt(norm(H) / n_training)

    # Compute the label data
    U = randn(n_training, m)
    for u in eachrow(U)
        u = u / norm(u)
    end
    return H, U
end

## Create the neural network
model = Flux.Chain(
    TinnitusReconstructor.TransformedDense(
        m => n, identity, cos; init=TinnitusReconstructor.scaled_uniform(; gain=2π)
    ),
)

## Create optimizer

opt_state = Flux.setup(AdamW(η, β, decay), model)

## Create data and batches


