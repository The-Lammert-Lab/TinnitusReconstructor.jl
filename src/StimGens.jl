#############################

## Type definitions

#############################
"Abstract supertype for all stimulus generation."
abstract type Stimgen end

"""
    BinnedStimgen <: Stimgen

Abstract supertype for all binned stimulus generation.
"""
abstract type BinnedStimgen <: Stimgen end

struct UniformPrior <: BinnedStimgen
    min_freq::Real
    max_freq::Real
    duration::Real
    n_trials::Int
    Fs::Real
    n_bins::Int
    min_bins::Int
    max_bins::Int

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
        @assert min_freq <= max_freq "`min_freq` must be less than `max_freq`"
        @assert min_bins < max_bins "`min_bins` cannot be greater than `max_bins`"
        @assert max_bins < n_bins "`max_bins` cannot be greater than `n_bins`"
        return new(min_freq, max_freq, duration, n_trials, Fs, n_bins, min_bins, max_bins)
    end
end

"""
    UniformPrior(; kwargs...) <: BinnedStimgen

Outer constructor for stimulus generation type in which 
    the number of filled bins is selected from 
    the Uniform distribution on the interval `[min_bins, max_bins]`.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `n_trials::Integer = 100`: The number of trials the subject will complete.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
- `n_bins::Integer = 100`: The number of bins into which to partition the frequency range.
- `min_bins::Integer = 10`: The minimum number of bins that may be filled on any stimuli.
- `max_bins::Integer = 50`: The maximum number of bins that may be filled on any stimuli.
"""
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
get_fs(s::Stimgen)::Int = s.Fs
get_nfft(s::Stimgen)::Int = get_fs(s) * s.duration

# Universal functions
function subject_selection_process(
    s::Stimgen, target_signal::AbstractVector{T}
) where {T<:Real}
    _, _, spect, binned_repr = generate_stimuli_matrix(s)
    e = spect'target_signal
    y = -ones(Int, size(e))
    y[e .>= quantile(e, 0.5; alpha=0.5, beta=0.5)] .= 1
    return y, spect, binned_repr
end

# Convert target_signal to Vector if passed as an Array.
function subject_selection_process(
    s::Stimgen, target_signal::AbstractMatrix{T}
) where {T<:Real}
    @assert size(target_signal, 2) == 1 "Target signal must be a Vector or single-column Matrix."
    return subject_selection_process(s, vec(target_signal))
end

"""
    synthesize_audio(X, nfft)

Synthesize audio from spectrum, X
"""
function synthesize_audio(X, nfft)
    phase = 2π * (rand(nfft ÷ 2) .- 0.5) # Assign random phase to freq spec
    s = @.. (10^(X / 10)) * cis(phase) # Convert dB to amplitudes
    ss = vcat(1, s)
    return irfft(ss, 2 * length(ss) - 1) #transform from freq to time domain
end

"""
    generate_stimuli_matrix(s::Stimgen)

Generate a stimuli matrix based on specifications in the stimgen type.
"""
function generate_stimuli_matrix(s::Stimgen)
    # Generate first stimulus
    stim, Fs, spect, _ = generate_stimulus(s)

    # Instantiate stimuli matrix
    stimuli_matrix = zeros(length(stim), s.n_trials)
    spect_matrix = zeros(Int, length(spect), s.n_trials)
    stimuli_matrix[:, 1] = stim
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

"""
    freq_bins(s::BinnedStimgen)

Generates a vector indicating which frequencies belong to the same bin,
    following a tonotopic map of audible frequency perception.
"""
@memoize function freq_bins(s::BinnedStimgen)
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
    binnum = zeros(Int, nfft ÷ 2)
    frequency_vector = collect(range(0, Fs ÷ 2, nfft ÷ 2))

    # This is a slow point
    for i in 1:(s.n_bins)
        @.. binnum[(frequency_vector <= binnd[i]) & (frequency_vector >= binst[i])] = i
    end

    return binnum, Fs, nfft, frequency_vector
end

function generate_stimuli_matrix(s::BinnedStimgen)
    # Generate first stimulus
    binned_repr_matrix = zeros(Int, s.n_bins, s.n_trials)
    stim, Fs, spect, binned_repr_matrix[:, 1] = generate_stimulus(s)

    # Instantiate stimuli matrix
    stimuli_matrix = zeros(length(stim), s.n_trials)
    spect_matrix = zeros(Int, length(spect), s.n_trials)
    stimuli_matrix[:, 1] = stim
    spect_matrix[:, 1] = spect
    for ii in 2:(s.n_trials)
        stimuli_matrix[:, ii], _, spect_matrix[:, ii], binned_repr_matrix[:, ii] = generate_stimulus(
            s
        )
    end

    return stimuli_matrix, Fs, spect_matrix, binned_repr_matrix
end

"""
    empty_spectrum(s::BinnedStimgen)

Generate an `nfft x 1` vector of Ints, where all values are -100. 
"""
empty_spectrum(s::BinnedStimgen) = -100 * ones(Int, get_nfft(s) ÷ 2)

"""
    spect2binnedrepr(s::BinnedStimgen, spect::AbstractArray{T}) where {T}

Get the binned representation of the spectrum.
 
Returns a vector containing the amplitude of the spectrum in each frequency bin.
"""
function spect2binnedrepr(s::BinnedStimgen, spect::AbstractArray{T}) where {T}
    binned_repr = zeros(s.n_bins, size(spect, 2))
    B, = freq_bins(s)

    @assert length(spect) == length(B)

    for bin_num in 1:(s.n_bins)
        a = spect[B .== bin_num, :]
        binned_repr[bin_num, :] .= a[1, :]
    end

    return binned_repr
end
