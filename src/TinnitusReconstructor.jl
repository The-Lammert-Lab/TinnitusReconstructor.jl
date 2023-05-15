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

export UniformPrior
export GaussianPrior
export Brimijoin
export Bernoulli
export BrimijoinGaussianSmoothed
export GaussianNoise
export UniformNoise
export GaussianNoiseNoBins
export UniformNoiseNoBins
export UniformPriorWeightedSampling
export PowerDistribution
export generate_stimuli_matrix
export generate_stimulus
export subject_selection_process
export cs
export gs
export spect2binnedrepr
export binnedrepr2spect
export wav2spect

end
