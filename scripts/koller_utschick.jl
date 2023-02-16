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
using ProgressMeter

## Dimensionality
# const m = 32
const m = 16
# const n = 128
const n = 32

## Hyperparameters

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

## Data parameters

# const n_training = 50_000
const n_training = 10_000
const n_val = 10_000

# sparsity
# const p = 5 # TODO
const p = 1

## Useful functions

"""
    loss(x, x̂)

Compute the loss function for this model.
"""
loss(x, x̂) = sum([TinnitusReconstructor.mmd(x, x̂, σ) for σ in σs])

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

## Create the neural network
model = Flux.Chain(
    TinnitusReconstructor.TransformedDense(
        n => m, identity, cos; init=TinnitusReconstructor.scaled_uniform(; gain=2π)
    ),
)

## Create optimizer

opt_state = Flux.setup(AdamW(η, β, decay), model)

## Create data and batches

H, U = generate_data(n_training, m, n, p)
data = DataLoader((H, U); batchsize=B)

## Training loop

losses = []
@showprogress 1 "loss" for (x, y) in data
    this_loss, grads = Flux.withgradient(model) do m
        ŷ = m(x)
        loss(y, ŷ)
    end
    Flux.update!(opt_state, model, grads[1])
    push!(losses, this_loss)
end

## TODO
# 1. Write signal to noise Function
# 2. Write OMP function.
# 3. Write testing loop that inputs noisy data with tunable noise parameter
#    and returns the signal to noise as a function of noise variance.
# 4. Implement hyperparameter tuning.
# 5. Show that we can generate a measurement matrix that satisfies constraints
#    and also has a high SNR.
