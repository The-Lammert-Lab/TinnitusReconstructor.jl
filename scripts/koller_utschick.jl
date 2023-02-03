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

## Parameters
const m = 10
const n = 2

## Create the neural network
model = Flux.Chain(
    TinnitusReconstructor.TransformedDense(
        m => n, identity, cos; init=TinnitusReconstructor.scaled_uniform(; gain=2π)
    ),
)
