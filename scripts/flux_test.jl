using Flux
# using TinnitusReconstructor
using Flux.Optimise: Adam, train!
using Flux.Data: DataLoader

model = Dense(16 => 100, identity; bias=false)
opt_state = Flux.setup(Adam(0.0001f0, (0.9f0, 0.999f0)), model)

data = (Float32.(rand(16, 100)), Float32.(ones(100, 100)))
dataloader = DataLoader(data, batchsize=4, shuffle=false)
loss(x, y) = Flux.Losses.mse(model(x), y)

@time train!(model, dataloader, opt_state) do m, x, y
    Flux.Losses.mse(m(x), y)
end