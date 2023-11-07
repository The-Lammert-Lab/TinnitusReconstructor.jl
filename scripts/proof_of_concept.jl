"""
Script that tests some core features.
"""

using TinnitusReconstructor
using DelimitedFiles
using FileIO
using Statistics

### Setup
stimgen = UniformPrior(; max_freq = 13e3, n_bins = 8, min_bins = 3, max_bins = 6)

Γ = 2
n_trials = 100

audio_file = "ATA/ATA_Tinnitus_Buzzing_Tone_1sec.wav"

audio = wav2spect(load(audio_file))

target_signal = 10 * log10.(audio)
binned_target_signal = spect2binnedrepr(stimgen, target_signal)

### Loop for statistics on randomly generated stimuli.
n = 30
r_loop_lr = zeros(n, 1)
r_loop_cs = zeros(n, 1)
for i in 1:n
    y, _, stim = subject_selection_process(stimgen, target_signal, n_trials)
    recon_lr = gs(y, stim')
    r = cor(recon_lr, binned_target_signal)
    r_loop_lr[i] = r[1]

    recon_cs = cs(y, stim', Γ)
    r = cor(recon_lr, binned_target_signal)
    r_loop_cs[i] = r[1]
end

r_mean_lr = mean(r_loop_lr)
r_std_lr = std(r_loop_lr)

r_mean_cs = mean(r_loop_cs)
r_std_cs = std(r_loop_cs)

### Print results
println("Mean linear r using a generated stimuli matrix for n = $n is $r_mean_lr")
println("Standard deviation of linear r using a generated stimuli matrix for n = $n is $r_std_lr")

println("----------")

println("Mean CS r using a generated stimuli matrix for n = $n is $r_mean_cs")
println("Standard deviation of CS r using a generated stimuli matrix for n = $n is $r_std_cs")
