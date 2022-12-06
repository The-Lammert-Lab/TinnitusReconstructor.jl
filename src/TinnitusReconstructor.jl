module TinnitusReconstructor

using FFTW
using LinearAlgebra
using Statistics
using PortAudio, SampledSignals
using FileIO
using LibSndFile
using DSP

include("funcs.jl")
include("StimGens.jl")
include("generate_stimuli.jl")

export UniformPrior
export present_stimulus, generate_stimuli_matrix

function present_stimulus(s::Stimgen)
    stimuli_matrix, Fs, _, _ = generate_stimuli_matrix(s)
    play_scaled_audio.(stimuli_matrix[:, 1], Fs)
    return nothing
end

end
