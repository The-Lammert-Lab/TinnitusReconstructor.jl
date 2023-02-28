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

"""
    UniformPrior(; kwargs...) <: BinnedStimgen

Constructor for stimulus generation type in which 
    the number of filled bins is selected from 
    the Uniform distribution on the interval `[min_bins, max_bins]`.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
- `n_bins::Integer = 100`: The number of bins into which to partition the frequency range.
- `min_bins::Integer = 10`: The minimum number of bins that may be filled on any stimuli.
- `max_bins::Integer = 50`: The maximum number of bins that may be filled on any stimuli.
"""
struct UniformPrior <: BinnedStimgen
    min_freq::Real
    max_freq::Real
    duration::Real
    Fs::Real
    n_bins::Int
    min_bins::Int
    max_bins::Int

    # Inner constructor to validate inputs
    function UniformPrior(;
        min_freq::R=100.0,
        max_freq::R=22e3,
        duration::R=0.5,
        Fs::R=44.1e3,
        n_bins::I=100,
        min_bins::I=10,
        max_bins::I=50,
    ) where {R<:Real,I<:Integer}
        @assert any(x -> x > 0, [min_freq max_freq duration Fs n_bins min_bins max_bins]) "All arguements must be greater than 0"
        @assert min_freq <= max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        @assert min_bins <= max_bins "`min_bins` cannot be greater than `max_bins`. `min_bins` = $min_bins, `max_bins` = $max_bins."
        @assert max_bins <= n_bins "`max_bins` cannot be greater than `n_bins`. `max_bins` = $max_bins, `n_bins` = $n_bins."
        return new(min_freq, max_freq, duration, Fs, n_bins, min_bins, max_bins)
    end
end

#############################

## Stimgen functions  

#############################

# Getter functions
@doc """
    fs(s::SG) where {SG<:Stimgen}

Return the number of samples per second.
"""
fs(s::SG) where {SG<:Stimgen} = s.Fs

@doc """
    nsamples(s::SG) where {SG<:Stimgen}

Return the number of samples as an Integer.
This means that the product `fs(s) * s.duration` must be an Integer
or an `InexactError` will be thrown.

# Examples
```jldoctest
julia> s = UniformPrior(;Fs=44.1e3, duration=0.5); nsamples(s)
22050

julia> s = UniformPrior(;Fs=44.1e3, duration=0.7); nsamples(s)
ERROR: InexactError: Int64(30869.999999999996)
"""
nsamples(s::SG) where {SG<:Stimgen} = convert(Int, fs(s) * s.duration)

# Universal functions
@doc """
    subject_selection_process(s::SG, target_signal::AbstractVector{T}, n_trials::I) where {SG<:Stimgen,T<:Real,I<:Integer}

Perform the synthetic subject decision process,
generating the stimuli on-the-fly using the stimulus
generation method `s`.
"""
function subject_selection_process(
    s::SG, target_signal::AbstractVector{T}, n_trials::I
) where {SG<:Stimgen,T<:Real,I<:Integer}
    _, _, spect, binned_repr = generate_stimuli_matrix(s, n_trials)
    e = spect'target_signal
    y = -ones(Int, size(e))
    y[e .>= quantile(e, 0.5; alpha=0.5, beta=0.5)] .= 1
    return y, spect, binned_repr
end

# Convert target_signal to Vector if passed as an Array.
@doc """
    subject_selection_process(s::SG, target_signal::AbstractMatrix{T}, n_trials::I) where {SG<:Stimgen,T<:Real,I<:Integer}
"""
function subject_selection_process(
    s::SG, target_signal::AbstractMatrix{T}, n_trials::I
) where {SG<:Stimgen,T<:Real,I<:Integer}
    return subject_selection_process(s, vec(target_signal), n_trials)
end

@doc """
    generate_stimuli_matrix(s::SG, n_trials::I) where {SG<:Stimgen, I<:Integer}

Generate `n_trials` of stimuli based on specifications in the stimgen type.

Returns `stimuli_matrix`, `Fs`, `spect_matrix`, `binned_repr_matrix`. 
    `binned_repr_matrix` = nothing if s >: BinnedStimgen.
"""
function generate_stimuli_matrix(s::SG, n_trials::I) where {SG<:Stimgen,I<:Integer}
    @assert n_trials > 1 "`n_trials` must be greater than 1. To generate one trial, use `generate_stimulus()`."

    # Generate first stimulus
    stim, Fs, spect, _ = generate_stimulus(s)

    # Instantiate stimuli matrix
    stimuli_matrix = zeros(length(stim), n_trials)
    spect_matrix = zeros(Int, length(spect), n_trials)
    stimuli_matrix[:, 1] = stim
    spect_matrix[:, 1] = spect
    for ii in 2:n_trials
        stimuli_matrix[:, ii], _, spect_matrix[:, ii], _ = generate_stimulus(s)
    end
    binned_repr_matrix = nothing

    return stimuli_matrix, Fs, spect_matrix, binned_repr_matrix
end

@doc """
    generate_stimuli_matrix(s::BS, n_trials::I) where {BS<:BinnedStimgen, I<:Integer}

Generate `n_trials` of stimuli based on specifications in the stimgen type.

Returns `stimuli_matrix`, `Fs`, `spect_matrix`, `binned_repr_matrix`. 
"""
function generate_stimuli_matrix(s::BS, n_trials::I) where {BS<:BinnedStimgen,I<:Integer}
    @assert n_trials > 1 "`n_trials` must be greater than 1. To generate one trial, use `generate_stimulus()`."

    # Generate first stimulus
    binned_repr_matrix = zeros(Int, s.n_bins, n_trials)
    stim, Fs, spect, binned_repr_matrix[:, 1] = generate_stimulus(s)

    # Instantiate stimuli matrix
    stimuli_matrix = zeros(length(stim), n_trials)
    spect_matrix = zeros(Int, length(spect), n_trials)
    stimuli_matrix[:, 1] = stim
    spect_matrix[:, 1] = spect
    for ii in 2:n_trials
        stimuli_matrix[:, ii], _, spect_matrix[:, ii], binned_repr_matrix[:, ii] = generate_stimulus(
            s
        )
    end

    return stimuli_matrix, Fs, spect_matrix, binned_repr_matrix
end

#############################

## BinnedStimgen functions  

#############################

@doc """
    freq_bins(s::BS) where {BS<:BinnedStimgen}

Generates a vector indicating which frequencies belong to the same bin,
    following a tonotopic map of audible frequency perception.
"""
@memoize function freq_bins(s::BS) where {BS<:BinnedStimgen}
    Fs = fs(s)
    nfft = nsamples(s)

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

@doc """
    empty_spectrum(s::BS) where {BS<:BinnedStimgen}

Generate an `nfft x 1` vector of Ints, where all values are -100. 
"""
empty_spectrum(s::BS) where {BS<:BinnedStimgen} = -100 * ones(Int(nsamples(s) ÷ 2))

@doc """
    spect2binnedrepr(s::BinnedStimgen, spect::AbstractArray{T}) where {BS<:BinnedStimgen,T}

Convert a spectral representation into a binned representation.
 
Returns an `n_trials x n_bins` array containing the amplitude of the spectrum in each frequency bin,
    where `n_trials` = size(binned_repr, 2).
@doc """
function spect2binnedrepr(s::BS, spect::AbstractArray{T}) where {BS<:BinnedStimgen,T}
    binned_repr = zeros(s.n_bins, size(spect, 2))
    B, = freq_bins(s)

    @assert length(spect) == length(B)

    for bin_num in 1:(s.n_bins)
        a = spect[B .== bin_num, :]
        binned_repr[bin_num, :] .= a[1, :]
    end

    return binned_repr
end

@doc """
    binnedrepr2spect(s::BinnedStimgen, binned_repr::AbstractArray{T}) where {BS<:BinnedStimgen,T}

Convert the binned representation into a spectral representation.

Returns an `n_frequencies x n_trials` spectral array, where `n_trials` = size(binned_repr, 2).
"""
function binnedrepr2spect(s::BS, binned_repr::AbstractArray{T}) where {BS<:BinnedStimgen,T}
    B, = freq_bins(s)
    spect = -100 * ones(length(B), size(binned_repr, 2))

    for bin_num in 1:(s.n_bins)
        spect[B .== bin_num, :] .= repeat(binned_repr[[bin_num], :], sum(B .== bin_num), 1)
    end

    return spect
end

#############################

## generate_stimulus functions  

#############################

# UniformPrior
@doc """
    generate_stimulus(s::UniformPrior)

Generate one stimulus sound.

Returns waveform, sample rate, spectral representation, binned representation, and a frequency vector.  
"""
function generate_stimulus(s::UniformPrior)
    # Define Frequency Bin Indices 1 through self.n_bins
    binnum, Fs, nfft, frequency_vector = freq_bins(s)
    spect = empty_spectrum(s)

    # sample from uniform distribution to get the number of bins to fill
    n_bins_to_fill = rand((s.min_bins):(s.max_bins))
    bins_to_fill = sample(1:(s.n_bins), n_bins_to_fill; replace=false)

    # Set spectrum ranges corresponding to bins to 0dB.
    [spect[binnum .== bins_to_fill[i]] .= 0 for i in 1:n_bins_to_fill]

    # Synthesize Audio
    stim = synthesize_audio(spect, nfft)

    # get the binned representation
    binned_repr = -100 * ones(s.n_bins)
    binned_repr[bins_to_fill] .= 0

    return stim, Fs, spect, binned_repr, frequency_vector
end
