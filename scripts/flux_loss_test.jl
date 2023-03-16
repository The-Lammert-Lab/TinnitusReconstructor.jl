using Flux
using Flux.Optimise: Adam, train!
using Flux.Data: DataLoader
using TinnitusReconstructor
using LinearAlgebra

function train_loop()
    model = Dense(16 => 100, identity; bias=false)
    opt_state = Flux.setup(Adam(0.0001f0, (0.9f0, 0.999f0)), model)
    H, U = TinnitusReconstructor.generate_data(100, 100, 16, 3)
    dataloader = DataLoader((H, U), batchsize=4)

    @time train!(model, dataloader, opt_state) do m, x, y
        this_mmd_loss = TinnitusReconstructor.mmd_loss(model(x), y; σs=[2, 5, 10, 20, 40, 80])
        this_l1_loss = 1f0 * norm(TinnitusReconstructor.invdB.(model.weight), 1)
        this_mmd_loss + this_l1_loss
    end
end

train_loop()