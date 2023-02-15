using TinnitusReconstructor
using Statistics
using Test

@testset "TinnitusReconstructor.jl" begin
    stimgen = UniformPrior(; min_bins=30, max_bins=30, max_freq=13e3)
    n_trials = 2000

    audio_file = "../ATA/ATA_Tinnitus_Buzzing_Tone_1sec.wav"
    audio = wav2spect(audio_file)
    target_signal = 10 * log10.(audio)

    binned_target_signal = spect2binnedrepr(stimgen, target_signal)

    responses, _, stim = subject_selection_process(stimgen, target_signal, n_trials)
    recon = gs(responses, stim')
    r = cor(recon, binned_target_signal)

    stimuli_matrix, Fs, spect_matrix, binned_repr_matrix = generate_stimuli_matrix(
        stimgen, n_trials
    )

    @test size(binned_target_signal) == (stimgen.n_bins, 1)
    @test r[1] >= 0.75

    @test size(stimuli_matrix, 2) == n_trials
    @test size(spect_matrix, 2) == n_trials
    @test Fs == stimgen.Fs
    @test size(binned_repr_matrix) == (stimgen.n_bins, n_trials)
end;
