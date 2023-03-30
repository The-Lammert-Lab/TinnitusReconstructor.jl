module TinnitusReconstructor

using FFTW
using LinearAlgebra
using Statistics, StatsBase
using PortAudio, SampledSignals
using FileIO
using LibSndFile
using DSP
using Memoize
using FastBroadcast
using Distributions: Uniform, DiscreteUniform, Normal, truncated, pdf
using Interpolations
using DelimitedFiles: writedlm, readdlm

include("funcs.jl")
include("StimGens.jl")

export UniformPrior, GaussianPrior
export BrimijoinGaussianSmoothed, Brimijoin
export Bernoulli, BrimijoinGaussianSmoothed
export GaussianNoise, UniformNoise
export GaussianNoiseNoBins, UniformNoiseNoBins
export UniformPriorWeightedSampling
export PowerDistribution
export build_distribution
export get_freq
export present_stimulus
export play_scaled_audio
export generate_stimuli_matrix
export generate_stimulus
export freq_bins
export spect2binnedrepr, binnedrepr2spect, wav2spect
export subject_selection_process
export cs, gs
export nsamples, fs, mels2hz, hz2mels
export empty_spectrum
export synthesize_audio
export crop_signal, crop_signal!

function present_stimulus(s::Stimgen)
    stimuli_matrix, Fs, _, _ = generate_stimuli_matrix(s)
    play_scaled_audio.(stimuli_matrix[:, 1], Fs)
    return nothing
end

end
