"""
This script trains a neural network
whose weights are a measurement matrix.
It then uses those weights as stimuli
for a simulated tinnitus reconstruction experiment
using the synthetic subject.
"""

# push!(LOAD_PATH, "..")

using Optimisers
using Zygote
using TinnitusReconstructor
using LinearAlgebra
using StatsBase: sample
using MLUtils
using ProgressMeter
using FileIO
using Statistics
using Flux: early_stopping, throttle
using BSON: @save
using Dates

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

# %% Paths
const tinnitus_sound_paths = (
    "../ATA/ATA_Tinnitus_Buzzing_Tone_1sec.wav", "../ATA/ATA_Tinnitus_Roaring_Tone_1sec.wav"
)

# %% Useful functions

@doc raw"""
    dB(x)

Convert from amplitude-scale to decibel-scale via

``\mathrm{dB}(x) = 10 \mathrm{log10}(x)``

# Examples
```jldoctest

julia> TinnitusReconstructor.dB.([1, 2, 100])
3-element Vector{Float64}:
  0.0
  3.010299956639812
 20.0
````

"""
dB(x) = 10log10(x)

@doc raw"""
    invdB(x)

Convert from decibel-scale to amplitude-scale via

``\mathrm{invdB}(x) = 10^{x/10}``

# Examples
```jldoctest
julia> TinnitusReconstructor.invdB.([-100, 0, 1, 2, 100])
5-element Vector{Float64}:
 1.0e-10
 1.0
 1.2589254117941673
 1.5848931924611136
 1.0e10
```

# See also
* [`dB`](@ref)
* [`db⁻¹`](@ref)
"""
invdB(x) = 10^(x / 10)

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

* [mmd](@ref mmd)
"""
function mmd_loss(x, x̂; σs=[1])
    return sum(mmd(x, x̂, σ) for σ in σs)
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
    return W * x
end

function load_test_data(n_bins, tinnitus_sound_paths)
    stimgen = UniformPrior(; n_bins=n_bins, min_freq=100., max_freq=13e3, min_bins=1, max_bins=1)
    target_signals = hcat(
        [dB.(wav2spect(audio_path)) for audio_path in tinnitus_sound_paths]...
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

function test(w::AbstractMatrix, s::SG, target_signals::AbstractMatrix, binned_target_signals::AbstractMatrix, p::Int) where SG <: TinnitusReconstructor.Stimgen
    W, target_signals, binned_target_signals = promote(W, target_signals, binned_target_signals)
    test(W, s, target_signals, binned_target_signals, p)
end

function create_es_cb(n_bins, tinnitus_sound_paths, p)
    stimgen, target_signals, binned_target_signals = load_test_data(
        n_bins, tinnitus_sound_paths
    )
    acc = let v = 0
        (W) -> v = test(W, stimgen, target_signals, binned_target_signals, p)
    end

    # Create early stopping callback
    es = early_stopping(acc, 10; distance=(best_score, score) -> score - best_score)
    return es
end

"""
    evalcb(model, opt_state, loss, acc, λ) -> str

Create a model name and save the model, loss, accuracy, and hyperparameters.
"""
function _evalcb(model, opt_state, loss, acc, λ)
    loss = round(loss; digits=3)
    acc = round(acc; digits=3)
    @info loss
    @info acc
    model_name = "model-date=$(now())-loss=$(loss)-acc=$(acc)-lambda=$(λ).bson"
    @save model_name model opt_state loss acc λ
    @info "model saved to $model_name"
    return model_name
end

"""
    create_eval_cb(timeout=1800)

Create a throttled callback that saves the model, loss, accuracy, and hyperparameters
"""
function create_eval_cb(timeout=1800)
    return throttle(
        timeout
    ) do W, stimgen, target_signals, binned_target_signals, p, model, opt_state, loss, λ
        acc = test(W, stimgen, target_signals, binned_target_signals, p)
        _evalcb(model, opt_state, loss, acc, λ)
    end
end

function main()
    # %% Create the training data
    H, U = generate_data(32, n_trials, n_bins, p)
    dataloader = MLUtils.DataLoader((H, U); batchsize=B)

    # Create test data
    stimgen, target_signals, binned_target_signals = load_test_data(n_bins, tinnitus_sound_paths)

    # Instantiate parameters
    W = rand(Float32, n_trials, n_bins)
    opt_state = Optimisers.setup(Optimisers.Adam(η, β), W)

    # Callbacks
    es = create_es_cb(n_bins, tinnitus_sound_paths, p)
    eval_cb = create_eval_cb()

    # Main loop

    ProgressMeter.@showprogress for (h, u) in dataloader
        # Zygote.gradient(W -> loss(model(h, W), u), W)
        L, Δ = Zygote.withgradient(W) do W
            this_mmd_loss = mmd_loss(model(h, W), u; σs=σs)
            this_l1_loss = λ * norm(invdB.(W), 1)
        end

        Optimisers.update(opt_state, W, Δ[1])
        @info L

        # Callbacks
        eval_cb(
            W,
            stimgen,
            target_signals,
            binned_target_signals,
            p,
            model,
            opt_state,
            L,
            λ,
        )
        if es()
            # Final steps
            acc = test(W, stimgen, target_signals, binned_target_signals, p)
            _evalcb(model, opt_state, L, acc, λ)
        end

    end
    
    @info "DONE!"

    # return save("weights.jld2", abs.(W))
end

main()
