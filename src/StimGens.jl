using StatsBase: mean

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

"""
    UniformPrior(; kwargs...) <: BinnedStimgen

Stimulus generation type in which 
    the number of filled bins is selected from 
    the Uniform distribution on the interval `[min_bins, max_bins]`.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
- `min_bins::Integer = 10`: The minimum number of bins that may be filled on any stimuli.
- `max_bins::Integer = 50`: The maximum number of bins that may be filled on any stimuli.
- `n_bins::Integer = 100`: The number of bins into which to partition the frequency range.
"""
struct UniformPrior{T <: Real, I <: Integer} <: BinnedStimgen
    min_freq::T
    max_freq::T
    duration::T
    Fs::T
    min_bins::I
    max_bins::I
    n_bins::I

    # Inner constructor to validate inputs
    function UniformPrior{T, I}(min_freq::T, max_freq::T, duration::T, Fs::T, min_bins::I,
            max_bins::I, n_bins::I) where {T <: Real, I <: Integer}
        @assert all(x -> x > 0, [min_freq max_freq duration Fs n_bins min_bins max_bins]) "All arguments must be greater than 0"
        @assert min_freq<=max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        @assert min_bins<=max_bins "`min_bins` cannot be greater than `max_bins`. `min_bins` = $min_bins, `max_bins` = $max_bins."
        @assert max_bins<=n_bins "`max_bins` cannot be greater than `n_bins`. `max_bins` = $max_bins, `n_bins` = $n_bins."
        @assert duration>0 "`duration` must be positive. `duration` = $duration."
        @assert isinteger(Fs * duration) "The product of `Fs` and `duration` (the number of samples) must be an integer."
        return new(min_freq, max_freq, duration, Fs, min_bins, max_bins, n_bins)
    end
end

function UniformPrior(;
        min_freq::Real = 100.0,
        max_freq::Real = 22e3,
        duration::Real = 0.5,
        Fs::Real = 44.1e3,
        min_bins::Integer = 10,
        max_bins::Integer = 50,
        n_bins::Integer = 100)
    reals = promote(min_freq, max_freq, duration, Fs)
    ints = promote(min_bins, max_bins, n_bins)
    return UniformPrior{eltype(reals), eltype(ints)}(reals..., ints...)
end

#####################################################

"""
    GaussianPrior(; kwargs...) <: BinnedStimgen

Stimulus generation type in which 
    the number of filled bins is selected from 
    from a Gaussian distribution with known mean and variance parameters.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
- `n_bins_filled_var::Real = 1`: The variance of number of bins that may be filled on any stimuli.
- `n_bins_filled_mean::Integer = 20`: The mean number of bins that may be filled on any stimuli.
- `n_bins::Integer = 100`: The number of bins into which to partition the frequency range.
"""
struct GaussianPrior{T <: Real, I <: Integer} <: BinnedStimgen
    min_freq::T
    max_freq::T
    duration::T
    Fs::T
    n_bins_filled_var::T
    n_bins_filled_mean::I
    n_bins::I

    # Inner constructor to validate inputs
    function GaussianPrior{T, I}(min_freq::T,
            max_freq::T,
            duration::T,
            Fs::T,
            n_bins_filled_var::T,
            n_bins_filled_mean::I,
            n_bins::I) where {T <: Real, I <: Integer}
        @assert all(x -> x > 0,
            [min_freq max_freq duration Fs n_bins n_bins_filled_mean n_bins_filled_var]) "All arguments must be greater than 0"
        @assert min_freq<=max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        @assert duration>0 "`duration` must be positive. `duration` = $duration."
        @assert isinteger(Fs * duration) "The product of `Fs` and `duration` (the number of samples) must be an integer."
        return new(min_freq, max_freq, duration, Fs, n_bins_filled_var, n_bins_filled_mean,
            n_bins)
    end
end

function GaussianPrior(;
        min_freq::Real = 100.0,
        max_freq::Real = 22e3,
        duration::Real = 0.5,
        Fs::Real = 44.1e3,
        n_bins_filled_var::Real = 1,
        n_bins_filled_mean::Integer = 20,
        n_bins::Integer = 100)
    reals = promote(min_freq, max_freq, duration, Fs, n_bins_filled_var)
    ints = promote(n_bins_filled_mean, n_bins)
    return GaussianPrior{eltype(reals), eltype(ints)}(reals..., ints...)
end

#####################################################

"""
    Bernoulli(; kwargs...) <: BinnedStimgen

Stimulus generation type in which 
    each tonotopic bin has a probability `bin_prob`
    of being at 0 dB, otherwise it is at $unfilled_db dB.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
- `bin_prob::Real=0.3`: The probability of a bin being filled.
- `n_bins::Integer = 100`: The number of bins into which to partition the frequency range.
"""
struct Bernoulli{T <: Real, I <: Integer} <: BinnedStimgen
    min_freq::T
    max_freq::T
    duration::T
    Fs::T
    bin_prob::T
    n_bins::I

    # Inner constructor to validate inputs
    function Bernoulli{T, I}(min_freq::T, max_freq::T, duration::T, Fs::T, bin_prob::T,
            n_bins::I) where {T <: Real, I <: Integer}
        @assert all(x -> x > 0, [min_freq max_freq duration Fs n_bins bin_prob]) "All arguments must be greater than 0"
        @assert min_freq<=max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        @assert bin_prob<=1 "`bin_prob` must be less than or equal to 1."
        @assert duration>0 "`duration` must be positive. `duration` = $duration."
        @assert isinteger(Fs * duration) "The product of `Fs` and `duration` (the number of samples) must be an integer."
        return new(min_freq, max_freq, duration, Fs, bin_prob, n_bins)
    end
end

function Bernoulli(;
        min_freq::Real = 100.0,
        max_freq::Real = 22e3,
        duration::Real = 0.5,
        Fs::Real = 44.1e3,
        bin_prob::Real = 0.3,
        n_bins::Integer = 100)
    reals = promote(min_freq, max_freq, duration, Fs, bin_prob)
    return Bernoulli{eltype(reals), typeof(n_bins)}(reals..., n_bins)
end

#####################################################

"""
    Brimijoin(; kwargs...) <: BinnedStimgen

Stimulus generation type in which 
    each tonotopic bin is filled with an amplitude 
    value from an equidistant list with equal probability.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
- `amp_min::Real = -20`: The lowest dB value a bin can have.
- `amp_max::Real = 0`: The highest dB value a bin can have.
- `amp_step::Int = 6`: The number of evenly spaced steps between `amp_min` and `amp_max`. 
- `n_bins::Integer = 100`: The number of bins into which to partition the frequency range.
"""
struct Brimijoin{T <: Real, I <: Integer} <: BinnedStimgen
    min_freq::T
    max_freq::T
    duration::T
    Fs::T
    amp_min::T
    amp_max::T
    amp_step::I
    n_bins::I

    # Inner constructor to validate inputs
    function Brimijoin{T, I}(min_freq::T,
            max_freq::T,
            duration::T,
            Fs::T,
            amp_min::T,
            amp_max::T,
            amp_step::I,
            n_bins::I) where {T <: Real, I <: Integer}
        @assert all(x -> x > 0, [min_freq max_freq duration Fs n_bins]) "Only amplitude arguments can be less than 0."
        @assert min_freq<=max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        @assert amp_min<amp_max "`amp_min` must be less than `amp_max`."
        @assert amp_step>1 "`amp_step` must be greater than 1."
        @assert duration>0 "`duration` must be positive. `duration` = $duration."
        @assert isinteger(Fs * duration) "The product of `Fs` and `duration` (the number of samples) must be an integer."
        return new(min_freq, max_freq, duration, Fs, amp_min, amp_max, amp_step, n_bins)
    end
end

function Brimijoin(;
        min_freq::Real = 100.0,
        max_freq::Real = 22e3,
        duration::Real = 0.5,
        Fs::Real = 44.1e3,
        amp_min::Real = -20,
        amp_max::Real = 0,
        amp_step::Integer = 6,
        n_bins::Integer = 100)
    reals = promote(min_freq, max_freq, duration, Fs, amp_min, amp_max)
    ints = promote(amp_step, n_bins)
    return Brimijoin{eltype(reals), eltype(ints)}(reals..., ints...)
end

#####################################################

"""
    BrimijoinGaussianSmoothed(; kwargs...) <: BinnedStimgen

Stimulus generation type in which 
    each tonotopic bin is filled by a Gaussian 
    with a maximum amplitude value chosen
    from an equidistant list with equal probability.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
- `amp_min::Real = -20`: The lowest dB value a bin can have.
- `amp_max::Real = 0`: The highest dB value a bin can have.
- `amp_step::Int = 6`: The number of evenly spaced steps between `amp_min` and `amp_max`. 
- `n_bins::Integer = 100`: The number of bins into which to partition the frequency range.
"""
struct BrimijoinGaussianSmoothed{T <: Real, I <: Integer} <: BinnedStimgen
    min_freq::T
    max_freq::T
    duration::T
    Fs::T
    amp_min::T
    amp_max::T
    amp_step::I
    n_bins::I

    # Inner constructor to validate inputs
    function BrimijoinGaussianSmoothed{T, I}(min_freq::T,
            max_freq::T,
            duration::T,
            Fs::T,
            amp_min::T,
            amp_max::T,
            amp_step::I,
            n_bins::I) where {T <: Real, I <: Integer}
        @assert all(x -> x > 0, [min_freq max_freq duration Fs n_bins]) "Only amplitude arguments can be less than 0."
        @assert min_freq<=max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        @assert amp_min<amp_max "`amp_min` must be less than `amp_max`."
        @assert amp_step>1 "`amp_step` must be greater than 1."
        @assert duration>0 "`duration` must be positive. `duration` = $duration."
        @assert isinteger(Fs * duration) "The product of `Fs` and `duration` (the number of samples) must be an integer."
        return new(min_freq, max_freq, duration, Fs, amp_min, amp_max, amp_step, n_bins)
    end
end

function BrimijoinGaussianSmoothed(;
        min_freq::Real = 100.0,
        max_freq::Real = 22e3,
        duration::Real = 0.5,
        Fs::Real = 44.1e3,
        amp_min::Real = -20,
        amp_max::Real = 0,
        amp_step::Integer = 6,
        n_bins::Integer = 100)
    reals = promote(min_freq, max_freq, duration, Fs, amp_min, amp_max)
    ints = promote(amp_step, n_bins)
    return BrimijoinGaussianSmoothed{eltype(reals), eltype(ints)}(reals..., ints...)
end

#####################################################

"""
    GaussianNoise(; kwargs...) <: BinnedStimgen

Stimulus generation type in which 
    each tonotopic bin is filled with amplitude chosen from a Gaussian distribution.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
- `n_bins::Integer = 100`: The number of bins into which to partition the frequency range.
- `amplitude_mean::Real = -10`: The mean of the Gaussian. 
- `amplitude_var::Real = 3`: The variance of the Gaussian. 
"""
struct GaussianNoise{T <: Real, I <: Integer} <: BinnedStimgen
    min_freq::T
    max_freq::T
    duration::T
    Fs::T
    amplitude_mean::T
    amplitude_var::T
    n_bins::I

    # Inner constructor to validate inputs
    function GaussianNoise{T, I}(min_freq::T,
            max_freq::T,
            duration::T,
            Fs::T,
            amplitude_mean::T,
            amplitude_var::T,
            n_bins::I) where {T <: Real, I <: Integer}
        @assert all(x -> x > 0, [min_freq max_freq duration Fs n_bins]) "Only amplitude mean can be less than 0."
        @assert amplitude_var>=0 "`amplitude_var` cannot be less than 0."
        @assert min_freq<=max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        @assert duration>0 "`duration` must be positive. `duration` = $duration."
        @assert isinteger(Fs * duration) "The product of `Fs` and `duration` (the number of samples) must be an integer."
        return new(min_freq, max_freq, duration, Fs, amplitude_mean, amplitude_var, n_bins)
    end
end

function GaussianNoise(;
        min_freq::Real = 100.0,
        max_freq::Real = 22e3,
        duration::Real = 0.5,
        Fs::Real = 44.1e3,
        amplitude_mean::Real = -10,
        amplitude_var::Real = 3,
        n_bins::Integer = 100)
    reals = promote(min_freq, max_freq, duration, Fs, amplitude_mean, amplitude_var)
    return GaussianNoise{eltype(reals), typeof(n_bins)}(reals..., n_bins)
end

#####################################################

"""
    UniformNoise(; kwargs...) <: BinnedStimgen

Stimulus generation type in which 
    each tonotopic bin is filled with amplitude chosen from a Uniform distribution.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
- `n_bins::Integer = 100`: The number of bins into which to partition the frequency range.
"""
struct UniformNoise{T <: Real, I <: Integer} <: BinnedStimgen
    min_freq::T
    max_freq::T
    duration::T
    Fs::T
    n_bins::I

    # Inner constructor to validate inputs
    function UniformNoise{T, I}(min_freq::T, max_freq::T, duration::T, Fs::T,
            n_bins::I) where {T <: Real, I <: Integer}
        @assert all(x -> x > 0, [min_freq max_freq duration Fs n_bins]) "All arguments must be greater than 0."
        @assert min_freq<=max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        @assert duration>0 "`duration` must be positive. `duration` = $duration."
        @assert isinteger(Fs * duration) "The product of `Fs` and `duration` (the number of samples) must be an integer."
        return new(min_freq, max_freq, duration, Fs, n_bins)
    end
end

function UniformNoise(;
        min_freq::Real = 100.0,
        max_freq::Real = 22e3,
        duration::Real = 0.5,
        Fs::Real = 44.1e3,
        n_bins::Integer = 100)
    reals = promote(min_freq, max_freq, duration, Fs)
    return UniformNoise{eltype(reals), typeof(n_bins)}(reals..., n_bins)
end

#####################################################

"""
    UniformPriorWeightedSampling(; kwargs...) <: BinnedStimgen

Stimulus generation type in which 
    each tonotopic bin is filled from a uniform distribution on [`min_bins`, `max_bins`]
    but which bins are filled is determined by a non-uniform distribution.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
- `alpha_::Real = 1`: The tuning parameter that exponentiates the number of unique frequencies in each bin.
- `min_bins::Integer = 10`: The minimum number of bins that may be filled on any stimuli.
- `max_bins::Integer = 50`: The maximum number of bins that may be filled on any stimuli.
- `n_bins::Integer = 100`: The number of bins into which to partition the frequency range.
"""
struct UniformPriorWeightedSampling{
    T <: Real,
    I <: Integer,
    Q <: AbstractVecOrMat{<:Real}
} <:
       BinnedStimgen
    min_freq::T
    max_freq::T
    duration::T
    Fs::T
    alpha_::T
    min_bins::I
    max_bins::I
    n_bins::I
    bin_probs::Q

    # Inner constructor to validate inputs and create bin_probs
    function UniformPriorWeightedSampling{T, I, Q}(min_freq::T,
            max_freq::T,
            duration::T,
            Fs::T,
            alpha_::T,
            min_bins::I,
            max_bins::I,
            n_bins::I) where {T <: Real,
            I <: Integer,
            Q <:
            AbstractVecOrMat{<:Real
            }}
        @assert all(x -> x > 0,
            [min_freq max_freq duration Fs n_bins min_bins max_bins alpha_]) "All arguments must be greater than 0."
        @assert min_freq<=max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        @assert min_bins<=max_bins "`min_bins` cannot be greater than `max_bins`. `min_bins` = $min_bins, `max_bins` = $max_bins."
        @assert max_bins<=n_bins "`max_bins` cannot be greater than `n_bins`. `max_bins` = $max_bins, `n_bins` = $n_bins."
        @assert duration>0 "`duration` must be positive. `duration` = $duration."
        @assert isinteger(Fs * duration) "The product of `Fs` and `duration` (the number of samples) must be an integer."

        binnums, = freq_bins(new(min_freq, max_freq, duration, Fs, alpha_, min_bins,
            max_bins, n_bins))

        # Compute the bin occupancy, which is a `n_bins x 1` vector
        # which counts the number of unique frequencies in each bin.
        # This bin occupancy quantity is not related to which bins
        # are "filled".
        bin_occupancy = zeros(n_bins)
        [bin_occupancy[i] = sum(binnums .== i) for i in 1:n_bins]

        # Set `bin_probs` equal to the bin occupancy, exponentiated by `alpha_`.
        bin_occupancy .^= alpha_
        bin_probs = normalize(bin_occupancy)

        return new{T, I, typeof(bin_probs)}(min_freq, max_freq, duration, Fs, alpha_,
            min_bins, max_bins, n_bins, bin_probs)
    end
end

function UniformPriorWeightedSampling(;
        min_freq::Real = 100.0,
        max_freq::Real = 22e3,
        duration::Real = 0.5,
        Fs::Real = 44.1e3,
        alpha_::Real = 1,
        min_bins::Integer = 10,
        max_bins::Integer = 50,
        n_bins::Integer = 100)
    reals = promote(min_freq, max_freq, duration, Fs, alpha_)
    ints = promote(min_bins, max_bins, n_bins)

    return UniformPriorWeightedSampling{eltype(reals), eltype(ints), Vector{Real}}(
        reals...,
        ints...)
end

#####################################################

"""
    PowerDistribution(; kwargs...) <: BinnedStimgen

Stimulus generation type in which 
    the frequencies in each bin are sampled 
    from a power distribution learned
    from tinnitus examples.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
- `n_bins::Integer = 100`: The number of bins into which to partition the frequency range.
- `distribution_filepath::AbstractString=joinpath(@__DIR__, "distribution.csv")`: The filepath to the default power distribution from which stimuli are generated
"""
struct PowerDistribution{
    T <: Real, I <: Integer, Q <: AbstractVecOrMat{<:Real},
    S <: AbstractString
} <: BinnedStimgen
    min_freq::T
    max_freq::T
    duration::T
    Fs::T
    n_bins::I
    distribution_filepath::S
    distribution::Q

    # Inner constructor to validate inputs
    function PowerDistribution{T, I, Q, S}(min_freq::T, max_freq::T, duration::T, Fs::T,
            n_bins::I,
            distribution_filepath::S) where {T <: Real,
            I <: Integer,
            Q <:
            AbstractVecOrMat{
                <:Real
            },
            S <:
            AbstractString}
        @assert all(x -> x > 0, [min_freq max_freq duration Fs n_bins]) "Only amplitude arguments can be less than 0."
        @assert min_freq<=max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        @assert duration>0 "`duration` must be positive. `duration` = $duration."
        @assert isinteger(Fs * duration) "The product of `Fs` and `duration` (the number of samples) must be an integer."

        if isfile(distribution_filepath)
            distribution = readdlm(distribution_filepath, ',')
        else
            distribution = build_distribution(new(min_freq, max_freq, duration, Fs, n_bins))
        end

        return new{T, I, typeof(distribution), S}(min_freq, max_freq, duration, Fs, n_bins,
            distribution_filepath, distribution)
    end
end

function PowerDistribution(;
        min_freq::Real = 100.0,
        max_freq::Real = 22e3,
        duration::Real = 0.5,
        Fs::Real = 44.1e3,
        n_bins::Integer = 100,
        distribution_filepath::AbstractString = joinpath(@__DIR__,
            "distribution.csv"))
    reals = promote(min_freq, max_freq, duration, Fs)
    return PowerDistribution{
        eltype(reals), typeof(n_bins), Vector{Real},
        typeof(distribution_filepath)
    }(reals...,
        n_bins,
        distribution_filepath)
end

#####################################################

"""
    UniformNoiseNoBins(; kwargs...) <: Stimgen

Stimulus generation type in which 
    each frequency is chosen from a uniform distribution on [$unfilled_db, 0] dB.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
"""
struct UniformNoiseNoBins{T <: Real} <: Stimgen
    min_freq::T
    max_freq::T
    duration::T
    Fs::T

    # Inner constructor to validate inputs
    function UniformNoiseNoBins{T}(min_freq::T, max_freq::T, duration::T,
            Fs::T) where {T <: Real}
        @assert all(x -> x > 0, [min_freq max_freq duration Fs]) "All arguments must be greater than 0."
        @assert min_freq<=max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        @assert duration>0 "`duration` must be positive. `duration` = $duration."
        @assert isinteger(Fs * duration) "The product of `Fs` and `duration` (the number of samples) must be an integer."
        return new(min_freq, max_freq, duration, Fs)
    end
end

function UniformNoiseNoBins(;
        min_freq::Real = 100.0, max_freq::Real = 22e3,
        duration::Real = 0.5, Fs::Real = 44.1e3)
    vals = promote(min_freq, max_freq, duration, Fs)
    return UniformNoiseNoBins{eltype(vals)}(vals...)
end

#####################################################

"""
    GaussianNoiseNoBins(; kwargs...) <: Stimgen

Stimulus generation type in which 
    each frequency's amplitude is chosen according to a Gaussian distribution.

# Keywords

- `min_freq::Real = 100`: The minimum frequency in range from which to sample.
- `max_freq::Real = 22e3`: The maximum frequency in range from which to sample.
- `duration::Real = 0.5`: The length of time for which stimuli are played in seconds.
- `Fs::Real = 44.1e3`: The frequency of the stimuli in Hz.
- `amplitude_mean::Real = -10`: The mean of the Gaussian. 
- `amplitude_var::Real = 3`: The variance of the Gaussian. 
"""
struct GaussianNoiseNoBins{T <: Real} <: Stimgen
    min_freq::T
    max_freq::T
    duration::T
    Fs::T
    amplitude_mean::T
    amplitude_var::T

    # Inner constructor to validate inputs
    function GaussianNoiseNoBins{T}(min_freq::T, max_freq::T, duration::T, Fs::T,
            amplitude_mean::T, amplitude_var::T) where {T <: Real}
        @assert all(x -> x > 0, [min_freq max_freq duration Fs]) "Only amplitude mean can be less than 0."
        @assert amplitude_var>=0 "`amplitude_var` cannot be less than 0."
        @assert min_freq<=max_freq "`min_freq` cannot be greater than `max_freq`. `min_freq` = $min_freq, `max_freq` = $max_freq."
        @assert duration>0 "`duration` must be positive. `duration` = $duration."
        @assert isinteger(Fs * duration) "The product of `Fs` and `duration` (the number of samples) must be an integer."
        return new(min_freq, max_freq, duration, Fs, amplitude_mean, amplitude_var)
    end
end

function GaussianNoiseNoBins(;
        min_freq::Real = 100.0,
        max_freq::Real = 22e3,
        duration::Real = 0.5,
        Fs::Real = 44.1e3,
        amplitude_mean::Real = -10,
        amplitude_var::Real = 3)
    vals = promote(min_freq, max_freq, duration, Fs, amplitude_mean, amplitude_var)
    return GaussianNoiseNoBins{eltype(vals)}(vals...)
end

#############################

## Stimgen functions  

#############################

# Getter functions
"""
    fs(s::SG) where {SG<:Stimgen}

Return the number of samples per second.
"""
fs(s::SG) where {SG <: Stimgen} = s.Fs

"""
    nsamples(s::SG) where {SG<:Stimgen}

Return the number of samples as an Integer.
This means that the product `fs(s) * s.duration` must be an Integer
or an `InexactError` will be thrown.

# Examples
```jldoctest
julia> s = UniformPrior(; Fs=44.1e3, duration=0.5); nsamples(s)
22050

julia> s = UniformPrior(; Fs=44.1e3, duration=0.7); nsamples(s)
ERROR: InexactError: Int64(30869.999999999996)
"""
nsamples(s::SG) where {SG <: Stimgen} = convert(Int, fs(s) * s.duration)

# Universal functions
"""
    subject_selection_process(s::SG, target_signal::AbstractVector{T}, n_trials::I) where {SG<:Stimgen,T<:Real,I<:Integer}

Perform the synthetic subject decision process,
generating the stimuli on-the-fly using the stimulus
generation method `s`.
"""
function subject_selection_process(s::SG, target_signal::AbstractVector{T},
        n_trials::I) where {SG <: Stimgen, T <: Real,
        I <: Integer}
    _, _, spect, binned_repr = generate_stimuli_matrix(s, n_trials)
    e = spect'target_signal
    y = -ones(Int, size(e))
    y[e .>= quantile(e, 0.5; alpha = 0.5, beta = 0.5)] .= 1
    return y, spect, binned_repr
end

# NOTE: Add check for second col is empty.
"""
    subject_selection_process(s::SG, target_signal::AbstractMatrix{T}, n_trials::I) where {SG<:Stimgen,T<:Real,I<:Integer}

    Convert target_signal to Vector if passed as a Matrix.
"""
function subject_selection_process(s::SG, target_signal::AbstractMatrix{T},
        n_trials::I) where {SG <: Stimgen, T <: Real,
        I <: Integer}
    return subject_selection_process(s, vec(target_signal), n_trials)
end

"""
    generate_stimuli_matrix(s::SG, n_trials::I) where {SG<:Stimgen, I<:Integer}

Generate `n_trials` of stimuli based on specifications in the stimgen type.

Returns `stimuli_matrix`, `Fs`, `spect_matrix`, `binned_repr_matrix`. 
    `binned_repr_matrix` = nothing if s >: BinnedStimgen.
"""
function generate_stimuli_matrix(s::SG, n_trials::I) where {SG <: Stimgen, I <: Integer}
    @assert n_trials>1 "`n_trials` must be greater than 1. To generate one trial, use `generate_stimulus()`."

    # Generate first stimulus
    stim, Fs, spect, _ = generate_stimulus(s)

    # Instantiate stimuli matrix
    stimuli_matrix = zeros(length(stim), n_trials)
    spect_matrix = zeros(length(spect), n_trials)
    stimuli_matrix[:, 1] = stim
    spect_matrix[:, 1] = spect
    for ii in 2:n_trials
        stimuli_matrix[:, ii], _, spect_matrix[:, ii], _ = generate_stimulus(s)
    end
    binned_repr_matrix = nothing

    return stimuli_matrix, Fs, spect_matrix, binned_repr_matrix
end

"""
    generate_stimuli_matrix(s::BS, n_trials::I) where {BS<:BinnedStimgen, I<:Integer}

Generate `n_trials` of stimuli based on specifications in the stimgen type.

Returns `stimuli_matrix`, `Fs`, `spect_matrix`, `binned_repr_matrix`. 
"""
function generate_stimuli_matrix(s::BS,
        n_trials::I) where {BS <: BinnedStimgen, I <: Integer}
    @assert n_trials>1 "`n_trials` must be greater than 1. To generate one trial, use `generate_stimulus()`."

    # Generate first stimulus
    binned_repr_matrix = zeros(s.n_bins, n_trials)
    stim, Fs, spect, binned_repr_matrix[:, 1] = generate_stimulus(s)

    # Instantiate stimuli matrix
    stimuli_matrix = zeros(length(stim), n_trials)
    spect_matrix = zeros(length(spect), n_trials)
    stimuli_matrix[:, 1] = stim
    spect_matrix[:, 1] = spect
    for ii in 2:n_trials
        stimuli_matrix[:, ii], _, spect_matrix[:, ii], binned_repr_matrix[:, ii] = generate_stimulus(s)
    end

    return stimuli_matrix, Fs, spect_matrix, binned_repr_matrix
end

function get_freq(s::SG) where {SG <: Stimgen}
    return range(s.min_freq, s.max_freq, nsamples(s) ÷ 2)
end

#############################

## BinnedStimgen functions  

#############################

"""
    freq_bins(s::BS; n_bins::I = 0) where {BS<:BinnedStimgen, I<:Integer}

Generates a vector indicating which frequencies belong to the same bin,
    following a tonotopic map of audible frequency perception.
    The number of bins can be specified with the `n_bins` parameter. 
    `s.n_bins` is used by default.
"""
@memoize function freq_bins(s::BS; n_bins::I = 0) where {BS <: BinnedStimgen, I <: Integer}
    Fs = fs(s)
    nfft = nsamples(s)

    if n_bins < 2
        n_bins = s.n_bins
    end

    # Define Frequency Bin Indices 1 through n_bins
    bintops = round.(mels2hz.(collect(range(hz2mels.(s.min_freq), hz2mels.(s.max_freq),
        n_bins + 1))))
    bin_starts = bintops[1:(end - 1)]
    bin_stops = bintops[2:end]
    binnum = zeros(Int, nfft ÷ 2)
    frequency_vector = collect(range(0, Fs ÷ 2, nfft ÷ 2))

    # This is a slow point
    for i in 1:(n_bins)
        @.. binnum[(frequency_vector <= bin_stops[i]) & (frequency_vector >= bin_starts[i])] = i
    end

    return binnum, Fs, nfft, frequency_vector, bin_starts, bin_stops
end

"""
    empty_spectrum(s::BS) where {BS<:BinnedStimgen}

Generate an `nfft x 1` vector of Ints, where all values are $unfilled_db. 
"""
empty_spectrum(s::BS) where {BS <: BinnedStimgen} = unfilled_db * ones(Int(nsamples(s) ÷ 2))

"""
    spect2binnedrepr(s::BinnedStimgen, spect::AbstractVecOrMat{T}; n_bins::I = 0) where {BS<:BinnedStimgen,T,I<:Integer}

Convert a spectral representation into a binned representation. 
The number of bins can be specified with the `n_bins` parameter. 
`s.n_bins` is used by default.
 
Returns an `n_trials x n_bins` array containing the amplitude of the spectrum in each frequency bin,
    where `n_trials` = size(binned_repr, 2).
"""
function spect2binnedrepr(s::BS, spect::AbstractVecOrMat{T};
        n_bins::I = 0) where {BS <: BinnedStimgen, T, I <: Integer}
    if n_bins < 2
        n_bins = s.n_bins
    end

    binned_repr = zeros(n_bins, size(spect, 2))
    B, = freq_bins(s; n_bins = n_bins)

    @assert size(spect, 1) == length(B)

    for bin_num in 1:(n_bins)
        a = spect[B .== bin_num, :]
        binned_repr[bin_num, :] .= mean(a, dims = 1)
    end

    return binned_repr
end

"""
    binnedrepr2spect(s::BinnedStimgen, binned_repr::AbstractArray{T}) where {BS<:BinnedStimgen,T}

Convert the binned representation into a spectral representation. 
The number of bins can be specified with the `n_bins` parameter. 
`s.n_bins` is used by default.

Returns an `n_frequencies x n_trials` spectral array, where `n_trials` = size(binned_repr, 2).
"""
function binnedrepr2spect(s::BS,
        binned_repr::AbstractArray{T};
        n_bins::I = 0) where {BS <: BinnedStimgen, T, I <: Integer}
    if n_bins < 2
        n_bins = s.n_bins
    end

    B, = freq_bins(s; n_bins = n_bins)
    spect = unfilled_db * ones(length(B), size(binned_repr, 2))

    for bin_num in 1:(n_bins)
        spect[B .== bin_num, :] .= repeat(binned_repr[[bin_num], :], sum(B .== bin_num), 1)
    end

    return spect
end

"""
    binnedrepr2wav(s::BS, binned_repr::AbstractVecOrMat{T}, mult, binrange,
    new_n_bins::I = 256) where {BS <: BinnedStimgen, I <: Integer, T}

Converts a binned spectral representation into a waveform 
by enhancing the resolution (to `new_n_bins`) via interpolation, 
sharpening the peaks (`mult`), and rescaling the dynamic range (`binrange`).

Returns a waveform and the associated frequency spectrum.
"""
function binnedrepr2wav(s::BS, binned_repr::AbstractVecOrMat{T}, mult, binrange,
        new_n_bins::I = 256) where {BS <: BinnedStimgen, I <: Integer, T}
    @assert(new_n_bins>s.n_bins,
        "New number of bins must be greater than current number of bins")

    # Rescale to between 0 and 1
    rescale!(binned_repr)

    # Interpolate to new_n_bins
    bin_idx = 1:(s.n_bins)
    bin_idx2 = range(1, s.n_bins, new_n_bins)

    itp = interpolate(vec(binned_repr), BSpline(Cubic(Natural(OnGrid()))))
    itps = scale(itp, bin_idx)
    binned_repr = itps[bin_idx2]

    # Sharpen bins in interpolated spectrum
    b = [1, -2, 1]
    C = conv(binned_repr, b)[2:(end - 1)]
    C[[1, end]] .= 0
    C2 = conv(C, b)[2:(end - 1)]
    C2[[1, 2, end - 1, end]] .= 0
    binned_repr = @. binned_repr - (mult * (50^2) / 40) * C + (mult * (50^4) / 600) * C2
    binned_repr = binned_repr .- minimum(binned_repr)

    # Rescale dynamic range of audio signal by adjusting bin heights
    binned_repr = rescale(binned_repr) .* binrange

    spect = binnedrepr2spect(s, binned_repr; n_bins = new_n_bins)
    wav = synthesize_audio(spect, nsamples(s))

    return wav, spect
end

"""
    build_distribution(s::PowerDistribution; save_path::AbstractString=@__DIR__)

Builds the default power distribution from ATA tinnitus sample files.
    Saves the distribution as a vector in dB at save_path.
"""
function build_distribution(s::PowerDistribution; save_path::AbstractString = @__DIR__)
    @assert ispath(save_path) "`save_path` must be a valid path"

    ATA_files = readdir(joinpath(pkgdir(TinnitusReconstructor), "ATA"); join = true)
    freq_vec = get_freq(s)

    audio = load(pop!(ATA_files))
    Fs_file = convert(Int, samplerate(audio))

    y = float(vec(audio.data))
    rescale!(y)
    Y = fft(y) / length(y)
    freq_file = Fs_file / 2 * range(0, 1, Fs_file ÷ 2 + 1)
    pxx = abs.(Y[1:(Fs_file ÷ 2 + 1)])

    power_spectra = zeros(length(pxx), length(ATA_files) + 1)
    power_spectra[:, 1] = TinnitusReconstructor.dB.(pxx)

    for (ind, file) in enumerate(eachrow(ATA_files))
        audio = load(file[1])
        y = float(vec(audio.data))
        rescale!(y)
        Y = fft(y) / length(y)
        pxx = abs.(Y[1:(Fs_file ÷ 2 + 1)])

        power_spectra[:, ind] = TinnitusReconstructor.dB.(pxx)
    end

    spect = mean(power_spectra; dims = 2)

    # Interpolate (analagous to MATLAB's interp1(freq_file, spect, freq_vec, 'cubic');)
    distribution = zeros(length(freq_vec))
    itp = interpolate(vec(spect), BSpline(Cubic(Natural(OnGrid()))))
    intf = scale(itp, (freq_file,))
    distribution = intf[freq_vec]

    writedlm(joinpath(save_path, "distribution.csv"), distribution, ',')

    return distribution
end

#############################

## generate_stimulus methods  

#############################

"""
    generate_stimulus(s::Stimgen)

Generate one stimulus sound.

Returns waveform, sample rate, spectral representation, 
binned representation, and a frequency vector 
(the last two empty if s >: BinnedStimgen). 
Methods are specialized for each concrete subtype of Stimgen.
"""
function generate_stimulus end

# UniformPrior
function generate_stimulus(s::UniformPrior)
    # Define Frequency Bin Indices 1 through self.n_bins
    binnum, Fs, nfft, frequency_vector, _, _ = freq_bins(s)
    spect = empty_spectrum(s)

    # sample from uniform distribution to get the number of bins to fill
    n_bins_to_fill = rand(DiscreteUniform(s.min_bins, s.max_bins))
    bins_to_fill = sample(1:(s.n_bins), n_bins_to_fill; replace = false)

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
    d = truncated(Normal(s.n_bins_filled_mean, sqrt(s.n_bins_filled_var)), 1, s.n_bins)
    n_bins_to_fill = round(Int, rand(d))
    bins_to_fill = sample(1:(s.n_bins), n_bins_to_fill; replace = false)

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
    [spect[binnum .== i] .= binned_repr[i] for i in 1:(s.n_bins)]

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
    [spect[binnum .== i] .= binned_repr[i] for i in 1:(s.n_bins)]

    # Synthesize Audio
    stim = synthesize_audio(spect, nfft)

    return stim, Fs, spect, binned_repr, frequency_vector
end

# BrimijoinGaussianSmoothed
# TODO: Optimize
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
    d = Normal.(μ, σ)

    normal = similar(frequency_vector)

    for i in 1:(s.n_bins)
        # Create a normal distribution with the correct number of points
        normal .= pdf.(d[i], frequency_vector)
        # Rescale
        normal .= binned_repr[i] * normal ./ maximum(normal)
        # Add to the spectrum
        spect += normal
    end

    rescale!(spect)
    spect = -unfilled_db .* (spect .- 1)

    # Synthesize Audio
    stim = synthesize_audio(spect, nfft)

    return stim, Fs, spect, binned_repr, frequency_vector
end

# GaussianNoise
function generate_stimulus(s::GaussianNoise)
    binnum, Fs, nfft, frequency_vector, _, _ = freq_bins(s)
    spect = empty_spectrum(s)

    # Get binned representation from random values of Gaussian distribution
    binned_repr = rand(Normal(s.amplitude_mean, sqrt(s.amplitude_var)), s.n_bins)

    # Set spectrum ranges corresponding to bin levels.
    [spect[binnum .== i] .= binned_repr[i] for i in 1:(s.n_bins)]

    # Synthesize Audio
    stim = synthesize_audio(spect, nfft)

    return stim, Fs, spect, binned_repr, frequency_vector
end

# UniformNoise
function generate_stimulus(s::UniformNoise)
    binnum, Fs, nfft, frequency_vector, _, _ = freq_bins(s)
    spect = empty_spectrum(s)

    # Get binned representation from random values of Uniform distribution
    binned_repr = rand(Uniform(unfilled_db, 0), s.n_bins)

    # Set spectrum ranges corresponding to bin levels.
    [spect[binnum .== i] .= binned_repr[i] for i in 1:(s.n_bins)]

    # Synthesize Audio
    stim = synthesize_audio(spect, nfft)

    return stim, Fs, spect, binned_repr, frequency_vector
end

# UniformNoiseNoBins
function generate_stimulus(s::UniformNoiseNoBins)
    Fs = fs(s)
    nfft = nsamples(s)

    # generate spectrum completely randomly without bins
    # amplitudes are uniformly-distributed between unfilled_db and 0.
    spect = rand(Uniform(unfilled_db, 0), nfft ÷ 2)

    # Synthesize Audio
    stim = synthesize_audio(spect, nfft)

    # Empty output
    binned_repr = []
    frequency_vector = []
    return stim, Fs, spect, binned_repr, frequency_vector
end

# GaussianNoiseNoBins
function generate_stimulus(s::GaussianNoiseNoBins)
    Fs = fs(s)
    nfft = nsamples(s)

    # generate spectrum completely randomly without bins
    # amplitudes are uniformly-distributed between unfilled_db and 0.
    spect = rand(Normal(s.amplitude_mean, sqrt(s.amplitude_var)), nfft ÷ 2)

    # Synthesize Audio
    stim = synthesize_audio(spect, nfft)

    # Empty output
    binned_repr = []
    frequency_vector = []
    return stim, Fs, spect, binned_repr, frequency_vector
end

# UniformPriorWeightedSampling
function generate_stimulus(s::UniformPriorWeightedSampling)
    binnum, Fs, nfft, frequency_vector, _, _ = freq_bins(s)
    spect = empty_spectrum(s)

    # Generate Random Freq Spec in dB Acccording to Frequency Bin Index

    # sample from uniform distribution to get the number of bins to fill
    n_bins_to_fill = rand(DiscreteUniform(s.min_bins, s.max_bins))

    # sample from a weighted distribution without replacement
    # to get the bins that should be filled
    filled_bins = sample(1:(s.n_bins), Weights(s.bin_probs), n_bins_to_fill;
        replace = false)

    # fill those bins
    [spect[binnum .== bin] .= 0 for bin in eachindex(filled_bins)]

    # Synthesize Audio
    stim = synthesize_audio(spect, nfft)

    # get the binned representation
    binned_repr = unfilled_db * ones(Int, s.n_bins)
    binned_repr[filled_bins] .= 0

    return stim, Fs, spect, binned_repr, frequency_vector
end

# PowerDistribution
function generate_stimulus(s::PowerDistribution)
    binnum, Fs, nfft, frequency_vector, _, _ = freq_bins(s)
    spect = empty_spectrum(s)

    # Get the histogram of the power distribution for binning
    # Force 16 bins using edges parameter. May be unnecessary.
    h = normalize(
        fit(Histogram, vec(s.distribution),
            range(extrema(s.distribution)..., 16));
        mode = :pdf)

    bin_centers = h.edges[1][1] .+ cumsum(diff(h.edges[1]) / 2)
    pdf = h.weights .+ (0.01 * mean(h.weights))
    pdf /= sum(pdf)
    cdf = cumsum(pdf)

    # Sample power values from the histogram
    r = rand(s.n_bins)
    binned_repr = zeros(s.n_bins)
    for i in eachindex(r)
        idx = argmin((cdf .- r[i]) .^ 2)
        binned_repr[i] = bin_centers[idx]
    end

    # Create the random frequency spectrum
    [spect[binnum .== i] .= binned_repr[i] for i in 1:(s.n_bins)]

    # Synthesize Audio
    stim = synthesize_audio(spect, nfft)

    return stim, Fs, spect, binned_repr, frequency_vector
end
