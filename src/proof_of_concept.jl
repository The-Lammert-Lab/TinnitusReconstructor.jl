"""
Script that loads the output of MATLAB wav2spect for 
    ATA_Tinnitus_Buzzing_Tone_1sec.wav (saved as a csv)
    and runs synthetic subject on it.

    Very janky, but intended only as a proof of concept.
"""

# TODO: Use Unitful to make conversion between spectrogram 
# output type and dB.

using TinnitusReconstructor
using DelimitedFiles
using Statistics

stimgen = UniformPrior(; max_freq=13e3, n_bins=8, min_bins=3, max_bins=6)

# Workaround until Julia function is fixed
s_f = readdlm("../wav2spect_output.csv", ',', Float64, '\n')
s = s_f[:, 1]
f = s_f[:, 2]

# To compare directly with MATLAB using same stimuli.
stimuli_matrix = readdlm("../stimuli_matrix_buzzing.csv", ',', Float64, '\n')

target_signal = 10 * log10.(s)
binned_target_signal = spect2binnedrepr(stimgen, target_signal)

# Works up to here

responses_synth, = subject_selection_process(binned_target_signal, stimuli_matrix')

reconstruction_synth = gs(responses_synth, stimuli_matrix')
r_static = cor(reconstruction_synth, binned_target_signal)
println("Linear r using set stimuli matrix = $(r_static[1])")

n = 500
r_loop = zeros(Float64, n, 1)
for i in 1:n
    y, _, stim = subject_selection_process(stimgen, target_signal)
    recon = gs(y, stim')
    r = cor(recon, binned_target_signal)
    r_loop[i] = r[1]
end

r_mean = mean(r_loop)
r_std = std(r_loop)

println("Mean r using a generated stimuli matrix for n = $n is $r_mean")
println("Standard deviation of r using a generated stimuli matrix for n = $n is $r_std")
