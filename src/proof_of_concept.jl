"""
Script that loads the output of MATLAB wav2spect for 
    ATA_Tinnitus_Buzzing_Tone_1sec.wav (saved as a csv)
    and runs synthetic subject on it.
"""

using TinnitusReconstructor
using DelimitedFiles
using FileIO
using Statistics

stimgen = UniformPrior(; max_freq=13e3, n_bins=8, min_bins=3, max_bins=6)

Gamma = 2

### Repeatable section with saved stimuli and spectral representation of Buzzing Tone.

# Because MATLAB and Julia wav2spect have numerically different outputs.
s_f = readdlm("../wav2spect_output.csv", ',', Float64, '\n')
s = s_f[:, 1]

# To compare directly with MATLAB using same stimuli.
stimuli_matrix = readdlm("../stimuli_matrix_buzzing.csv", ',', Float64, '\n')

target_signal = 10 * log10.(s)
binned_target_signal = spect2binnedrepr(stimgen, target_signal)

responses_synth, = subject_selection_process(stimuli_matrix', binned_target_signal)

reconstruction_synth_lr = gs(responses_synth, stimuli_matrix')
r_static_lr = cor(reconstruction_synth_lr, binned_target_signal)

reconstruction_synth_cs = cs(responses_synth, stimuli_matrix', Gamma)
r_static_cs = cor(reconstruction_synth_cs, binned_target_signal)

### Using Julia wav2spect.

audio_file = "ATA/ATA_Tinnitus_Buzzing_Tone_1sec.wav"

audio = wav2spect(audio_file)

target_signal = 10 * log10.(audio)
binned_target_signal = spect2binnedrepr(stimgen, target_signal)

responses_synth, = subject_selection_process(stimuli_matrix', binned_target_signal)

reconstruction_synth_lr = gs(responses_synth, stimuli_matrix')
r_Julia_lr = cor(reconstruction_synth_lr, binned_target_signal)

reconstruction_synth_cs = cs(responses_synth, stimuli_matrix', Gamma)
r_Julia_cs = cor(reconstruction_synth_cs, binned_target_signal)

### Loop for statistics on randomly generated stimuli.

n = 100
r_loop_lr = zeros(n, 1)
r_loop_cs = zeros(n, 1)
for i in 1:n
    y, _, stim = subject_selection_process(stimgen, target_signal)
    recon_lr = gs(y, stim')
    r = cor(recon_lr, binned_target_signal)
    r_loop_lr[i] = r[1]

    recon_cs = cs(y, stim', Gamma)
    r = cor(recon_lr, binned_target_signal)
    r_loop_cs[i] = r[1]
end

r_mean_lr = mean(r_loop_lr)
r_std_lr = std(r_loop_lr)

r_mean_cs = mean(r_loop_cs)
r_std_cs = std(r_loop_cs)

println("Linear r using read-in stimuli matrix and target signal = $(r_static_lr[1])")
println(
    "Linear r using read-in stimuli matrix and Julia wav2spect target signal = $(r_Julia_lr[1])",
)
println("Mean linear r using a generated stimuli matrix for n = $n is $r_mean_lr")
println(
    "Standard deviation of linear r using a generated stimuli matrix for n = $n is $r_std_lr",
)

println("----------")

println("CS r using read-in stimuli matrix and target signal = $(r_static_cs[1])")
println(
    "CS r using read-in stimuli matrix and Julia wav2spect target signal = $(r_Julia_cs[1])"
)
println("Mean CS r using a generated stimuli matrix for n = $n is $r_mean_cs")
println(
    "Standard deviation of CS r using a generated stimuli matrix for n = $n is $r_std_cs"
)
