#############################

## Type definitions

#############################

abstract type Stimgen end
abstract type BinnedStimgen <: Stimgen end

struct UniformPrior <: BinnedStimgen
    min_freq::Real
    max_freq::Real
    duration::Real
    n_trials::Integer
    Fs::Real
    n_bins::Integer
    min_bins::Integer
    max_bins::Integer

    # Inner constructor to validate inputs
    function UniformPrior(
        min_freq::Real,
        max_freq::Real,
        duration::Real,
        n_trials::Integer,
        Fs::Real,
        n_bins::Integer,
        min_bins::Integer,
        max_bins::Integer,
    )
        @assert any(
            x -> x >= 0, [min_freq max_freq duration n_trials Fs n_bins min_bins max_bins]
        ) "All arguements must be greater than 0"
        @assert min_freq >= max_freq "`min_freq` must be less than `max_freq`"
        @assert min_bins > max_bins "`min_bins` cannot be greater than `max_bins`"
        @assert max_bins > n_bins "`max_bins` cannot be greater than `n_bins`"
        return new(min_freq, max_freq, duration, n_trials, Fs, n_bins, min_bins, max_bins)
    end
end

# Outer constructor for default values
function UniformPrior(;
    min_freq=100,
    max_freq=22e3,
    duration=0.5,
    n_trials=100,
    Fs=44.1e3,
    n_bins=100,
    min_bins=10,
    max_bins=50,
)
    return UniformPrior(
        min_freq, max_freq, duration, n_trials, Fs, n_bins, min_bins, max_bins
    )
end

#############################

## Stimgen functions  

#############################

# Getter functions
get_fs(s::Stimgen)::Int64 = s.Fs
get_nfft(s::Stimgen)::Int64 = get_fs(s) * s.duration

# Universal functions
function subject_selection_process(s::Stimgen, signal)
    _, _, spect, binned_repr = generate_stimuli_matrix(s)
    e = spect'signal
    y = -ones(Int64, size(e))
    y[e .>= quantile(e, 0.5; alpha=0.5, beta=0.5)] .= 1
    return y, spect, binned_repr
end

"""
synthesize_audio(X, nfft)
Synthesize audio from spectrum, X
"""
function synthesize_audio(X, nfft)
    phase = 2π * (rand(nfft ÷ 2, 1) .- 0.5) # Assign random phase to freq spec
    s = @. (10^(X / 10)) * exp(phase * im) # Convert dB to amplitudes
    ss = vcat(1, s, conj(reverse(s; dims=1)))
    return real.(ifft(ss)) #transform from freq to time domain
end

function generate_stimuli_matrix(s::Stimgen)
    # Generate first stimulus
    stim1, Fs, spect, _ = generate_stimulus(s)

    # Instantiate stimuli matrix
    stimuli_matrix = zeros(Float64, length(stim1), s.n_trials)
    spect_matrix = zeros(Int64, length(spect), s.n_trials)
    stimuli_matrix[:, 1] = stim1
    spect_matrix[:, 1] = spect
    for ii in 2:(s.n_trials)
        stimuli_matrix[:, ii], _, spect_matrix[:, ii], _ = generate_stimulus(s)
    end
    binned_repr_matrix = nothing

    return stimuli_matrix, Fs, spect_matrix, binned_repr_matrix
end

#############################

## BinnedStimgen functions  

#############################

function freq_bins(s::BinnedStimgen)
    Fs = get_fs(s)
    nfft = get_nfft(s)

    # Define Frequency Bin Indices 1 through self.n_bins
    bintops =
        round.(
            mels2hz.(
                collect(range(hz2mels.(s.min_freq), hz2mels.(s.max_freq), s.n_bins + 1))
            )
        )
    binst = bintops[1:(end - 1)]
    binnd = bintops[2:end]
    binnum = zeros(Int64, nfft ÷ 2, 1)
    frequency_vector = collect(range(0, Fs ÷ 2, nfft ÷ 2))

    for i in 1:(s.n_bins)
        @. binnum[(frequency_vector <= binnd[i]) & (frequency_vector >= binst[i])] = i
    end

    return binnum, Fs, nfft, frequency_vector
end

function generate_stimuli_matrix(s::BinnedStimgen)
    # Generate first stimulus
    binned_repr_matrix = zeros(Int64, s.n_bins, s.n_trials)
    stim1, Fs, spect, binned_repr_matrix[:, 1] = generate_stimulus(s)

    # Instantiate stimuli matrix
    stimuli_matrix = zeros(Float64, length(stim1), s.n_trials)
    spect_matrix = zeros(Int64, length(spect), s.n_trials)
    stimuli_matrix[:, 1] = stim1
    spect_matrix[:, 1] = spect
    for ii in 2:(s.n_trials)
        stimuli_matrix[:, ii], _, spect_matrix[:, ii], binned_repr_matrix[:, ii] = generate_stimulus(
            s
        )
    end

    return stimuli_matrix, Fs, spect_matrix, binned_repr_matrix
end

empty_spectrum(s::BinnedStimgen) = -100 * ones(Int64, get_nfft(s) ÷ 2, 1)

function spect2binnedrepr(s::BinnedStimgen, T)
    binned_repr = zeros(s.n_bins, size(T, 2))
    B = freq_bins(s)

    @assert length(T) == length(B)

    for bin_num in 1:(s.n_bins)
        a = T[B .== bin_num, :]
        binned_repr[bin_num, :] .= a[1, :]
    end

    return binned_repr
end
