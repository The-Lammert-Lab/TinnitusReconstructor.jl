using TinnitusReconstructor
using Statistics
using Test

const n_bins = 80
const min_freq = 100
const max_freq = 13e3

const n_trials = 800

const BINNED_STIMGEN = [
    UniformPrior(;
        n_bins=n_bins, max_freq=max_freq, min_freq=min_freq, min_bins=30, max_bins=30
    ),
    GaussianPrior(;
        n_bins=n_bins,
        max_freq=max_freq,
        min_freq=min_freq,
        n_bins_filled_mean=20,
        n_bins_filled_var=0.01,
    ),
    Brimijoin(; n_bins=n_bins, max_freq=max_freq, min_freq=min_freq),
    Bernoulli(; n_bins=n_bins, max_freq=max_freq, min_freq=min_freq, bin_prob=0.5),
    BrimijoinGaussianSmoothed(; n_bins=n_bins, max_freq=max_freq, min_freq=min_freq),
    GaussianNoise(;
        n_bins=n_bins,
        max_freq=max_freq,
        min_freq=min_freq,
        amplitude_mean=-35,
        amplitude_var=5,
    ),
    UniformNoise(; n_bins=n_bins, max_freq=max_freq, min_freq=min_freq),
    UniformPriorWeightedSampling(;
        n_bins=n_bins,
        max_freq=max_freq,
        min_freq=min_freq,
        min_bins=3,
        max_bins=10,
        alpha_=1,
    ),
    PowerDistribution(; n_bins=n_bins, max_freq=max_freq, min_freq=min_freq, distribution_filepath="."),
]

const UNBINNED_STIMGEN = [GaussianNoiseNoBins(), UniformNoiseNoBins()]

@testset showtiming = true "Stimgen: $(typeof(BINNED_STIMGEN[i]))" for i in eachindex(
    BINNED_STIMGEN
)
    stimgen = BINNED_STIMGEN[i]
    audio_file = "../ATA/ATA_Tinnitus_Buzzing_Tone_1sec.wav"
    audio = wav2spect(audio_file)
    target_signal = 10 * log10.(audio)

    binned_target_signal = spect2binnedrepr(stimgen, target_signal)

    responses, _, stim = subject_selection_process(stimgen, target_signal, n_trials)
    recon_linear = gs(responses, stim')
    recon_cs = cs(responses, stim')
    r_linear = cor(recon_linear, binned_target_signal)
    r_cs = cor(recon_cs, binned_target_signal)

    stimuli_matrix, Fs, spect_matrix, binned_repr_matrix = generate_stimuli_matrix(
        stimgen, n_trials
    )

    @test size(binned_target_signal) == (stimgen.n_bins, 1)
    @test r_linear[1] >= 0.7
    @test r_cs[1] >= 0.7

    @test size(stimuli_matrix, 2) == n_trials
    @test size(spect_matrix, 2) == n_trials
    @test Fs == stimgen.Fs
    @test size(binned_repr_matrix) == (stimgen.n_bins, n_trials)
end

@testset "Stimgen: $(typeof(UNBINNED_STIMGEN[i]))" for i in eachindex(UNBINNED_STIMGEN)
    stimgen = UNBINNED_STIMGEN[i]

    stimuli_matrix, Fs, spect_matrix, _ = generate_stimuli_matrix(stimgen, n_trials)

    @test size(stimuli_matrix, 2) == n_trials
    @test size(spect_matrix, 2) == n_trials
    @test Fs == stimgen.Fs
end
