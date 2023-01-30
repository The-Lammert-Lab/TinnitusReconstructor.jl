"""
This script is a test to implement
the maximum mean discrepancy optimization
problem for designing a compressive sensing matrix
with structural constraints
by Koller & Utschick 2022.
"""

using Flux
using Flux.Data: DataLoader
using TinnitusReconstructor

function loss(x, y)
    TinnitusReconstructor.mmd(x, y)
end

"""
    create_model(m::T, n::T) where T <: Int

TBW
"""
function create_model(m::T, n::T) where T <: Int
    model = Chain(
        Dense(m => n, identity, bias=false),
        TinnitusReconstructor.phase_to_mm,
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
