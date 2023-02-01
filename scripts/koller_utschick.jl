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


function loss(x, y)
    TinnitusReconstructor.mmd(x, y)
end

@doc raw"""
    scaled_uniform([rng = Flux.default_rng_value()], size...; gain = 1) -> Array
    scaled_uniform([rng], kw...) -> Function

Return an `Array{Float32}` of the given `size` containing random numbers drawn from a uniform distribution
on the interval ``gain * [0, 1]``.

# Examples

julia> round.(extrema(scaled_uniform(100, 10; gain=2π)), digits=3)
(0.004, 6.282)

julia> scaled_uniform(5) |> summary
"5-element Vector{Float32}"
"""
function scaled_uniform(rng::AbstractRNG, dims::Integer...; gain::Real=1)
    (rand(rng, Float32, dims...)) .* gain * 1f0
end
scaled_uniform(dims::Integer...; kw...) = scaled_uniform(Flux.default_rng_value(), dims...; kw...)
scaled_uniform(rng::AbstractRNG=Flux.default_rng_value(); init_kwargs...) = (dims...; kwargs...) -> scaled_uniform(rng, dims...; init_kwargs..., kwargs...)

@doc raw"""
    create_model(m::T, n::T) where T <: Int

A model which implements the transform

`\mathrm{stk}(A(\Phi)h)`

given the real matrix `\Phi \in \mathbb{R}^{m \times n}`

"""
function create_model(m::T, n::T) where T <: Int
    model = Chain(
        Dense(m => n, identity, bias=false),
        TinnitusReconstructor.phase_to_mm,
        # multiply by h
        TinnitusReconstructor.stk
    )
end

function train!(model, data_loader, optim)
    losses = Float64[]
    for data in data_loader
        input, label = data

        this_loss, this_grads = Flux.withgradient(model) do m
            result = m(input)
            loss(result, label)
        end
    Flux.update!(optim, model, this_grads[1])
    push!(losses, this_loss)
    end

end

function main()
    m, n = 10, 2
    model = create_model(m, n)
    # TODO: training data
    data_loader = DataLoader(training_data, batchsize=32, shuffle=false)
    optim = Flux.setup(Flux.Adam(0.01), model)

    train!(model, data_loader, optim)
