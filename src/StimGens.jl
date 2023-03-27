#############################

## Type definitions

#############################
"dB level of unfilled bins"
const unfilled_db = -100

"Abstract supertype for all stimulus generation."
abstract type Stimgen end

"""
    BinnedStimgen <: Stimgen

Abstract supertype for all binned stimulus generation.
"""
abstract type BinnedStimgen <: Stimgen end

#####################################################

struct UniformPrior <: BinnedStimgen
    min_freq::Real
    max_freq::Real
    duration::Real
    Fs::Real
    n_bins::Int
    min_bins::Int
    max_bins::Int

    # Inner constructor to validate inputs
    function UniformPrior(
        min_freq::Real,
        max_freq::Real,
        duration::Real,
        Fs::Real,
        n_bins::Integer,
        min_bins::Integer,
        max_bins::Integer,
    )
        @assert all(x -> x > 0, [min_freq max_freq duration Fs n_bins min_bins max_bins]) "All arguments must be greater than 0"
        @assert min_freq <= max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        @assert min_bins <= max_bins "`min_bins` cannot be greater than `max_bins`. `min_bins` = $min_bins, `max_bins` = $max_bins."
        @assert max_bins <= n_bins "`max_bins` cannot be greater than `n_bins`. `max_bins` = $max_bins, `n_bins` = $n_bins."
        return new(min_freq, max_freq, duration, Fs, n_bins, min_bins, max_bins)
    end
end

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
function UniformPrior(;
    min_freq=100.0,
    max_freq=22e3,
    duration=0.5,
    Fs=44.1e3,
    n_bins=100,
    min_bins=10,
    max_bins=50,
)
    return UniformPrior(min_freq, max_freq, duration, Fs, n_bins, min_bins, max_bins)
end

#####################################################

struct GaussianPrior <: BinnedStimgen
    min_freq::Real
    max_freq::Real
    duration::Real
    Fs::Real
    n_bins::Int
    n_bins_filled_mean::Int
    n_bins_filled_var::Real

    # Inner constructor to validate inputs
    function GaussianPrior(
        min_freq::Real,
        max_freq::Real,
        duration::Real,
        Fs::Real,
        n_bins::Integer,
        n_bins_filled_mean::Integer,
        n_bins_filled_var::Real
    )
        @assert all(x -> x > 0, [min_freq max_freq duration Fs n_bins n_bins_filled_mean n_bins_filled_var]) "All arguments must be greater than 0"
        @assert min_freq <= max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        return new(min_freq, max_freq, duration, Fs, n_bins, n_bins_filled_mean, n_bins_filled_var)
    end
end

"""
    GaussianPrior(; kwargs...) <: BinnedStimgen

Constructor for stimulus generation type in which 
    the number of filled bins is selected from 
    from a Gaussian distribution with known mean and variance parameters.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
- `n_bins::Integer = 100`: The number of bins into which to partition the frequency range.
- `n_bins_filled_mean::Integer = 20`: The mean number of bins that may be filled on any stimuli.
- `n_bins_filled_var::Real = 1`: The variance of number of bins that may be filled on any stimuli.
"""
function GaussianPrior(;
    min_freq=100.0,
    max_freq=22e3,
    duration=0.5,
    Fs=44.1e3,
    n_bins=100,
    n_bins_filled_mean=20,
    n_bins_filled_var=1,
)
    return GaussianPrior(min_freq, max_freq, duration, Fs, n_bins, n_bins_filled_mean, n_bins_filled_var)
end

#####################################################

struct Bernoulli <: BinnedStimgen
    min_freq::Real
    max_freq::Real
    duration::Real
    Fs::Real
    n_bins::Int
    bin_prob::Real

    # Inner constructor to validate inputs
    function Bernoulli(
        min_freq::Real,
        max_freq::Real,
        duration::Real,
        Fs::Real,
        n_bins::Integer,
        bin_prob::Real,
    )
        @assert all(x -> x > 0, [min_freq max_freq duration Fs n_bins bin_prob]) "All arguments must be greater than 0"
        @assert min_freq <= max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        @assert bin_prob <= 1 "`bin_prob` must be less than or equal to 1."
        return new(min_freq, max_freq, duration, Fs, n_bins, bin_prob)
    end
end

"""
    Bernoulli(; kwargs...) <: BinnedStimgen

Constructor for stimulus generation type in which 
    in which each tonotopic bin has a probability `bin_prob`
    of being at 0 dB, otherwise it is at $(unfilled_db) dB.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
- `n_bins::Integer = 100`: The number of bins into which to partition the frequency range.
- `bin_prob::Real=0.3`: The probability of a bin being filled.
"""
function Bernoulli(;
    min_freq=100.0,
    max_freq=22e3,
    duration=0.5,
    Fs=44.1e3,
    n_bins=100,
    bin_prob=0.3,
)
    return Bernoulli(min_freq, max_freq, duration, Fs, n_bins, bin_prob)
end

#####################################################

struct Brimijoin <: BinnedStimgen
    min_freq::Real
    max_freq::Real
    duration::Real
    Fs::Real
    n_bins::Int
    amp_min::Real
    amp_max::Real
    amp_step::Int

    # Inner constructor to validate inputs
    function Brimijoin(
        min_freq::Real,
        max_freq::Real,
        duration::Real,
        Fs::Real,
        n_bins::Integer,
        amp_min::Real,
        amp_max::Real,
        amp_step::Integer,
    )
        @assert all(x -> x > 0, [min_freq max_freq duration Fs n_bins]) "Only amplitude arguments can be less than 0."
        @assert min_freq <= max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        @assert amp_min < amp_max "`amp_min` must be less than `amp_max`."
        @assert amp_step > 1 "`amp_step` must be greater than 1."
        return new(min_freq, max_freq, duration, Fs, n_bins, amp_min, amp_max, amp_step)
    end
end

"""
    Brimijoin(; kwargs...) <: BinnedStimgen

Constructor for stimulus generation type in which 
    in which each tonotopic bin is filled with an amplitude 
    value from an equidistant list with equal probability.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
- `n_bins::Integer = 100`: The number of bins into which to partition the frequency range.
- `amp_min::Real = -20`: The lowest dB value a bin can have.
- `amp_max::Real = 0`: The highest dB value a bin can have.
- `amp_step::Int = 6`: The number of evenly spaced steps between `amp_min` and `amp_max`. 
"""
function Brimijoin(;
    min_freq=100.0,
    max_freq=22e3,
    duration=0.5,
    Fs=44.1e3,
    n_bins=100,
    amp_min=-20,
    amp_max=0,
    amp_step=6,
)
    return Brimijoin(min_freq, max_freq, duration, Fs, n_bins, amp_min, amp_max, amp_step)
end

#####################################################

struct BrimijoinGaussianSmoothed <: BinnedStimgen
    min_freq::Real
    max_freq::Real
    duration::Real
    Fs::Real
    n_bins::Int
    amp_min::Real
    amp_max::Real
    amp_step::Int

    # Inner constructor to validate inputs
    function BrimijoinGaussianSmoothed(
        min_freq::Real,
        max_freq::Real,
        duration::Real,
        Fs::Real,
        n_bins::Integer,
        amp_min::Real,
        amp_max::Real,
        amp_step::Integer,
    )
        @assert all(x -> x > 0, [min_freq max_freq duration Fs n_bins]) "Only amplitude arguments can be less than 0."
        @assert min_freq <= max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        @assert amp_min < amp_max "`amp_min` must be less than `amp_max`."
        @assert amp_step > 1 "`amp_step` must be greater than 1."
        return new(min_freq, max_freq, duration, Fs, n_bins, amp_min, amp_max, amp_step)
    end
end

"""
    BrimijoinGaussianSmoothed(; kwargs...) <: BinnedStimgen

Constructor for stimulus generation type in which 
    in which each tonotopic bin is filled by a Gaussian 
    with a maximum amplitude value chosen
    from an equidistant list with equal probability.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
- `n_bins::Integer = 100`: The number of bins into which to partition the frequency range.
- `amp_min::Real = -20`: The lowest dB value a bin can have.
- `amp_max::Real = 0`: The highest dB value a bin can have.
- `amp_step::Int = 6`: The number of evenly spaced steps between `amp_min` and `amp_max`. 
"""
function BrimijoinGaussianSmoothed(;
    min_freq=100.0,
    max_freq=22e3,
    duration=0.5,
    Fs=44.1e3,
    n_bins=100,
    amp_min=-20,
    amp_max=0,
    amp_step=6,
)
    return BrimijoinGaussianSmoothed(min_freq, max_freq, duration, Fs, n_bins, amp_min, amp_max, amp_step)
end

#####################################################

struct GaussianNoise <: BinnedStimgen
    min_freq::Real
    max_freq::Real
    duration::Real
    Fs::Real
    n_bins::Int
    amplitude_mean::Real
    amplitude_var::Real

    # Inner constructor to validate inputs
    function GaussianNoise(
        min_freq::Real,
        max_freq::Real,
        duration::Real,
        Fs::Real,
        n_bins::Integer,
        amplitude_mean::Real,
        amplitude_var::Real,
    )
        @assert all(x -> x > 0, [min_freq max_freq duration Fs n_bins]) "Only amplitude mean can be less than 0."
        @assert amplitude_var >= 0 "`amplitude_var` cannot be less than 0."
        @assert min_freq <= max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        return new(min_freq, max_freq, duration, Fs, n_bins, amplitude_mean, amplitude_var)
    end
end

"""
    GaussianNoise(; kwargs...) <: BinnedStimgen

Constructor for stimulus generation type in which 
    in which each tonotopic bin is filled
    with amplitude chosen from a Gaussian distribution.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
- `n_bins::Integer = 100`: The number of bins into which to partition the frequency range.
- `amplitude_mean::Real = -10`: The mean of the Gaussian. 
- `amplitude_var::Real = 3`: The variance of the Gaussian. 
"""
function GaussianNoise(;
    min_freq=100.0,
    max_freq=22e3,
    duration=0.5,
    Fs=44.1e3,
    n_bins=100,
    amplitude_mean=-10,
    amplitude_var=3,
)
    return GaussianNoise(min_freq, max_freq, duration, Fs, n_bins, amplitude_mean, amplitude_var)
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
function subject_selection_process(
    s::SG, target_signal::AbstractMatrix{T}, n_trials::I
) where {SG<:Stimgen,T<:Real,I<:Integer}
    @assert size(target_signal, 2) == 1 "Target signal must be a Vector or single-column Matrix."
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
    bin_starts = bintops[1:(end - 1)]
    bin_stops = bintops[2:end]
    binnum = zeros(Int, nfft ÷ 2)
    frequency_vector = collect(range(0, Fs ÷ 2, nfft ÷ 2))

    # This is a slow point
    for i in 1:(s.n_bins)
        @.. binnum[(frequency_vector <= bin_stops[i]) & (frequency_vector >= bin_starts[i])] = i
    end

    return binnum, Fs, nfft, frequency_vector, bin_starts, bin_stops
end

@doc """
    empty_spectrum(s::BS) where {BS<:BinnedStimgen}

Generate an `nfft x 1` vector of Ints, where all values are $unfilled_db. 
"""
empty_spectrum(s::BS) where {BS<:BinnedStimgen} = unfilled_db * ones(Int(nsamples(s) ÷ 2))

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

@doc """
    generate_stimulus(s::Stimgen)

Generate one stimulus sound.

Returns waveform, sample rate, spectral representation, binned representation, and a frequency vector. Methods are specialized for each concrete subtype of Stimgen.
"""
function generate_stimulus end

# UniformPrior
function generate_stimulus(s::UniformPrior)
    # Define Frequency Bin Indices 1 through self.n_bins
    binnum, Fs, nfft, frequency_vector, _, _ = freq_bins(s)
    spect = empty_spectrum(s)

    # sample from uniform distribution to get the number of bins to fill
    n_bins_to_fill = rand(Distributions.DiscreteUniform(s.min_bins, s.max_bins))
    bins_to_fill = sample(1:(s.n_bins), n_bins_to_fill; replace=false)

    # Set spectrum ranges corresponding to bins to 0dB.
    [spect[binnum .== bins_to_fill[i]] .= 0 for i in 1:n_bins_to_fill]

    # Synthesize Audio
    stim = synthesize_audio(spect, nfft)

    # get the binned representation
    binned_repr = unfilled_db * ones(Int, s.n_bins)
    binned_repr[bins_to_fill] .= 0

    return stim, Fs, spect, binned_repr, frequency_vector
end

# GaussianPrior
function generate_stimulus(s::GaussianPrior)
    binnum, Fs, nfft, frequency_vector, _, _ = freq_bins(s)
    spect = empty_spectrum(s)

    # sample from gaussian distribution to get the number of bins to fill
    d = Distributions.truncated(Distributions.Normal(s.n_bins_filled_mean, sqrt(s.n_bins_filled_var)), 1, s.n_bins)
    n_bins_to_fill = round(rand(d))
    bins_to_fill = sample(1:(s.n_bins), n_bins_to_fill; replace=false)

    # Set spectrum ranges corresponding to bins to 0dB.
    [spect[binnum .== bins_to_fill[i]] .= 0 for i in 1:n_bins_to_fill]

    # Synthesize Audio
    stim = synthesize_audio(spect, nfft)

    # get the binned representation
    binned_repr = unfilled_db * ones(Int, s.n_bins)
    binned_repr[bins_to_fill] .= 0

    return stim, Fs, spect, binned_repr, frequency_vector
end

# Bernoulli
function generate_stimulus(s::Bernoulli)
    binnum, Fs, nfft, frequency_vector, _, _ = freq_bins(s)
    spect = empty_spectrum(s)

    # Get binned representation
    binned_repr = unfilled_db * ones(Int, s.n_bins)
    binned_repr[rand(s.n_bins) .< s.bin_prob] .= 0

    # Set spectrum ranges corresponding to bins to bin level.
    [spect[binnum .== i] .= binned_repr[i] for i in 1:s.n_bins]

    # Synthesize Audio
    stim = synthesize_audio(spect, nfft)

    return stim, Fs, spect, binned_repr, frequency_vector
end

# Brimijoin
function generate_stimulus(s::Brimijoin)
    binnum, Fs, nfft, frequency_vector, _, _ = freq_bins(s)
    spect = empty_spectrum(s)

    # Get binned representation by sampling with replacement
    binned_repr = sample(range(s.amp_min, s.amp_max, s.amp_step), s.n_bins)

    # Set spectrum ranges corresponding to bin levels.
    [spect[binnum .== i] .= binned_repr[i] for i in 1:s.n_bins]

    # Synthesize Audio
    stim = synthesize_audio(spect, nfft)

    return stim, Fs, spect, binned_repr, frequency_vector
end

# BrimijoinGaussianSmoothed
function generate_stimulus(s::BrimijoinGaussianSmoothed)
    _, Fs, nfft, frequency_vector, bin_starts, bin_stops = freq_bins(s)
    spect = empty_spectrum(s)

    # Get binned representation by sampling with replacement
    binned_repr = sample(range(s.amp_min, s.amp_max, s.amp_step), s.n_bins)

    # μ: the center of the bins
    μ = (bin_starts .+ bin_stops) ./ 2

    # σ: half the width of the bins
    σ = (bin_stops .- bin_starts) ./ 2

    # Create distributions
    d = Distributions.Normal.(μ, σ)
    
    for i in 1:s.n_bins
        # Create a normal distribution with the correct number of points
        normal = Distributions.pdf.(d[i], frequency_vector)
        # Rescale
        normal = binned_repr[i] * normal ./ maximum(normal)
        # Add to the spectrum
        spect += normal
    end
    
    spect = (spect.-minimum(spect))./(maximum(spect).-minimum(spect))
    spect = -unfilled_db.*(spect .- 1)

    # Synthesize Audio
    stim = synthesize_audio(spect, nfft)

    return stim, Fs, spect, binned_repr, frequency_vector
end

# GaussianNoise
function generate_stimulus(s::GaussianNoise)
    binnum, Fs, nfft, frequency_vector, _, _ = freq_bins(s)
    spect = empty_spectrum(s)

    # Get binned representation from random values of Gaussian distribution
    binned_repr = rand(Distributions.Normal(s.amplitude_mean, sqrt(s.amplitude_var)), s.n_bins)

    # Set spectrum ranges corresponding to bin levels.
    [spect[binnum .== i] .= binned_repr[i] for i in 1:s.n_bins]

    # Synthesize Audio
    stim = synthesize_audio(spect, nfft)

    return stim, Fs, spect, binned_repr, frequency_vector
end
