"""
This script trains a neural network
whose weights are a measurement matrix.
It then uses those weights as stimuli
for a simulated tinnitus reconstruction experiment
using the synthetic subject.
"""

# push!(LOAD_PATH, "..")
cd("/home/alec/code/TinnitusReconstructor.jl/scripts/")

using Flux
using Flux.Optimise: Adam, train!
using Flux.Data: DataLoader
using Random: MersenneTwister
using TinnitusReconstructor
using LinearAlgebra
using ProgressLogging
using Zygote

const rng = MersenneTwister(1234)

# %% Constants

const n_trials = 100
const n_bins = 16

# %% Hyperparameters

# learning rate
const η = 0.00001f0 # [1e-6, 5e-3]
# ADAM momementum
const β = (0.9f0, 0.999f0) # [0.9, 1] (K&U used ADAM not ADAMW)
# weight decay
const decay = 0.0f0
# Gaussian kernel standard deviation
const σs = [2, 5, 10, 20, 40, 80]
# Batch size
const B = 16 # [150, 1500]
# L1 loss coefficient
const λ = 0.001f0

# %% Data parameters

const n_training = 5_000
const n_val = 10_000

# sparsity
const p = 3

# %% Paths
const tinnitus_sound_paths = (
    "../ATA/ATA_Tinnitus_Buzzing_Tone_1sec.wav", "../ATA/ATA_Tinnitus_Roaring_Tone_1sec.wav"
)

function load_test_data(n_bins, tinnitus_sound_paths)
    stimgen = UniformPrior(; n_bins=n_bins, min_freq=100., max_freq=13e3, min_bins=1, max_bins=1)
    target_signals = hcat(
        [TinnitusReconstructor.dB.(wav2spect(audio_path)) for audio_path in tinnitus_sound_paths]...
    )

    binned_target_signals = spect2binnedrepr(stimgen, target_signals)
    return stimgen, target_signals, binned_target_signals
end

function test(
    W::AbstractMatrix{T},
    s::SG,
    target_signals::AbstractMatrix{T2},
    binned_target_signals::AbstractMatrix{T2},
    p::Int,
) where {SG<:TinnitusReconstructor.Stimgen,T<:Real,T2<:Real}
    # Map the stimuli from bin-space to frequency-space

    nts = size(target_signals, 2)
    stimuli_matrix = binnedrepr2spect(s, W')
    # Perform the experiment with the synthetic subject
    r = 0.0
    for i in 1:nts
        y, _, _ = subject_selection_process(stimuli_matrix, target_signals[:, i])
        x = cs(y, W, p)
        r += cor(x, binned_target_signals[:, i])
    end
    return r / nts
end

function test(W::AbstractMatrix, s::SG, target_signals::AbstractMatrix, binned_target_signals::AbstractMatrix, p::Int) where SG <: TinnitusReconstructor.Stimgen
    W, target_signals, binned_target_signals = promote(W, target_signals, binned_target_signals)
    test(W, s, target_signals, binned_target_signals, p)
end

function train_loop(η, λ)
    model = Dense(n_bins, n_trials, identity; bias=false)
    opt_state = Flux.setup(Adam(η, β), model)
    H, U = TinnitusReconstructor.generate_data(10000, n_trials, n_bins, p)
    dataloader = DataLoader((H, U), batchsize=B, parallel=true)

    # @withprogress train!(model, dataloader, opt_state) do m, x, y
    #     this_mmd_loss = TinnitusReconstructor.mmd_loss(m(x), y; σs=[2, 5, 10, 20, 40, 80])
    #     this_l1_loss = λ * norm(TinnitusReconstructor.invdB.(m.weight), 1)
    #     this_mmd_loss + this_l1_loss
    # end

    for (i, (h, u)) in enumerate(dataloader)
        # Zygote.gradient(W -> loss(model(h, W), u), W)
        L, Δ = Zygote.withgradient(model) do m
            this_mmd_loss = TinnitusReconstructor.mmd_loss(m(h), u; σs=σs)
            this_l1_loss = λ * norm(TinnitusReconstructor.invdB.(m.weight), 1)
            this_mmd_loss + this_l1_loss
        end

	    opt_state, model = Flux.update!(opt_state, model, Δ[1]) 
        @info "$i of $(length(dataloader))"
        @info "loss = $(round(L; digits=3))"
    end

end

# """
#     evalcb(model, opt_state, loss, acc, λ) -> str

# Create a model name and save the model, loss, accuracy, and hyperparameters.
# """
# function _evalcb(model, opt_state, loss, acc, λ, η)
#     loss = round(loss; digits=3)
#     acc = round(acc; digits=3)
#     @info "loss = $loss"
#     @info "acc = $acc"
#     @info "lr = $η"
#     model_name = "model-date=$(now())-loss=$(loss)-acc=$(acc)-lambda=$(λ)-lr=$(η).bson"
#     @save model_name model opt_state loss acc λ
#     @info "model saved to $model_name"
#     return model_name, acc
# end

# """
#     create_eval_cb(timeout=1800)

# Create a throttled callback that saves the model, loss, accuracy, and hyperparameters
# """
# function create_eval_cb(timeout=1800)
#     return throttle(
#         timeout
#     ) do W, stimgen, target_signals, binned_target_signals, p, model, opt_state, loss, λ, η
#         acc = test(W, stimgen, target_signals, binned_target_signals, p)
#         _evalcb(model, opt_state, loss, acc, λ, η)
#     end
# end

# function train_loop(η, λ)
#     # %% Create the training data
#     H, U = generate_data(100, n_trials, n_bins, p)
#     dataloader = MLUtils.DataLoader((H, U); batchsize=B, parallel=true)

#     # Create test data
#     stimgen, target_signals, binned_target_signals = load_test_data(n_bins, tinnitus_sound_paths)

#     # Instantiate parameters
#     W = randn(rng, Float32, n_trials, n_bins)
#     opt_state = Optimisers.setup(Optimisers.Adam(η, β), W)

#     # Callbacks
#     eval_cb = create_eval_cb()

#     # Main loop
#     patience_counter = 0
#     best_loss = Inf

#     for (i, (h, u)) in enumerate(dataloader)
#         # Zygote.gradient(W -> loss(model(h, W), u), W)
#         L, Δ = Zygote.withgradient(W) do W
#             # this_mmd_loss = mmd_loss(model(h, W), u; σs=σs)
#             # this_l1_loss = λ * norm(invdB.(W), 1)
#             # this_mmd_loss + this_l1_loss
#             Flux.Losses.mse(model(h, W), u)
#         end

#         #opt_state, W = Optimisers.update(opt_state, W, Δ[1])
# 	    opt_state, W = Optimisers.update!(opt_state, W, Δ[1]) 
#         @info "$i of $(length(dataloader))"
#         @info "loss = $(round(L; digits=3))"

#         # # Callbacks
#         # _, acc = eval_cb(
#         #     W,
#         #     stimgen,
#         #     target_signals,
#         #     binned_target_signals,
#         #     p,
#         #     model,
#         #     opt_state,
#         #     L,
#         #     λ,
#         #     η
#         # )
        
#         # # early stopping
#         # if L < best_loss
#         #     best_loss = L
#         #     patience_counter = 0
#         # else
#         #     patience_counter += 1
#         # end
#         # if patience_counter > 100
#         #     _evalcb(W, opt_state, L, acc, λ, η)
#         #     @info "early stopping triggered"
#         #     break
#         # end

#     end
    
#     @info "DONE!"

#     # return save("weights.jld2", abs.(W))
# end

function main()
    # ηs = Float32.(10. .^ [-1, -2, -3, -4])
    # λs = Float32.(10. .^ [0, -1, -2, -3])
    # @showprogress for (η, λ) in product(ηs, λs)
    #     train_loop(η, λ)
    # end
    train_loop(1f-2, 1f0)
end

@time main()
