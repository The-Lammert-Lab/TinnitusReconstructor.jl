import DSP: stft
using FileIO: load
import LibSndFile

hz2mels(f) = 2595 * log10(1 + (f / 700))

mels2hz(m) = 700 * (10^(m / 2595) - 1)

"""
    synthesize_audio(spect::AbstractVecOrMat, nfft::Integer)

Synthesize audio from a frequency spectrum `spect` using a number of FFT points `nfft`.

# Arguments
- `spect::AbstractVecOrMat`: frequency spectrum
- `nfft::Integer`: number of FFT points
"""
function synthesize_audio(spect::AbstractVecOrMat, nfft::Integer)
    phase = 2π * (rand(nfft ÷ 2) .- 0.5) # Assign random phase to freq spec
    s = @.. (10^(spect / 10)) * cis(phase) # Convert dB to amplitudes
    ss = vcat(1, s)
    return irfft(ss, 2 * size(ss, 1) - 1) #transform from freq to time domain
end

"""
    soft_threshold(α, γ)

Soft thresholding operator for use in compressed sensing.
"""
function soft_threshold(α, γ)
    β = zeros(length(α))

    ind = @. abs(α) > γ

    @. β[ind] = @views sign(α[ind]) * (abs(α[ind]) - γ)

    return β
end

"""
    zhangpassivegamma(Φ::AbstractArray{T}, y, Γ::Integer) where {T<:Real}

Passive algorithm for 1-bit compressed sensing with no basis.

# References
- Zhang, L., Yi, J. and Jin, R., 2014, June. Efficient algorithms for robust one-bit compressive sensing. In *International Conference on Machine Learning* (pp. 820-828). PMLR.
"""
function zhangpassivegamma(Φ::AbstractArray{T}, y, Γ::Integer) where {T <: Real}
    m = size(Φ, 1)
    n = size(Φ, 2)

    α = (1 / m) * (Φ'y)

    val = sort(abs.(α); rev = true)
    γ = val[Γ + 1]

    if norm(α, Inf) <= γ
        return zeros(n, 1)
    else
        return (1 / norm(soft_threshold(α, γ), 2)) * soft_threshold(α, γ)
    end
end

"""
    gs(responses, Φ)

Linear reverse correlation.

# References
- Gosselin, F. and Schyns, P.G., 2003. Superstitious perceptions reveal properties of internal representations. *Psychological science*, 14(5), pp.505-509.
"""
gs(responses, Φ) = (1 / size(Φ, 2)) * Φ'responses

"""
    cs(responses, Φ, Γ::Integer=32)

One-bit compressed sensing reverse correlation with basis.

# References
- Plan, Y. and Vershynin, R., 2012. Robust 1-bit compressed sensing and sparse logistic regression: A convex programming approach. *IEEE Transactions on Information Theory*, 59(1), pp.482-494.

- Zhang, L., Yi, J. and Jin, R., 2014, June. Efficient algorithms for robust one-bit compressive sensing. In *International Conference on Machine Learning* (pp. 820-828). PMLR.
"""
function cs(responses, Φ::AbstractArray{T}, Γ::Integer = 32) where {T <: Real}
    n_samples = length(responses)
    len_signal = size(Φ, 2)

    ek = zeros(Int, len_signal)
    p = plan_idct!(ek)

    Θ = zeros(n_samples, len_signal)
    for i in 1:len_signal
        ek = zeros(Int, len_signal)
        ek[i] = 1
        Ψ = p * ek
        Θ[:, i] .= Φ * Ψ
    end

    s = zhangpassivegamma(Θ, responses, Γ)

    x = zeros(len_signal)
    for i in 1:len_signal
        ek = zeros(Int, len_signal)
        ek[i] = 1
        Ψ = p * ek
        x .+= Ψ * s[i]
    end

    return x
end

cs_no_basis(Φ, responses, Γ = 32) = zhangpassivegamma(Φ, responses, Γ)

"""
    subject_selection_process(stimuli_matrix::AbstractVecOrMat{T}, target_signal::AbstractVector{T}) where {T<:Real}

Perform the synthetic subject decision process, given a matrix of precomputed stimuli `stimuli_matrix`
and a `target_signal`.
The `stimuli_matrix` is of size `m x n` where `m` is the number of trials and `n` is the number of samples in the signal.
The `target_signal` is a flat vector of size `n` or an `n x 1` matrix.
Return the `n`-dimensional response vector `y` as well as the `stimuli_matrix`
as well as `nothing` for the binned representation.
"""
function subject_selection_process(stimuli_matrix::AbstractVecOrMat{T},
        target_signal::AbstractVector{T}) where {T <: Real}
    e = stimuli_matrix'target_signal
    y = -ones(Int, size(e))
    y[e .>= quantile(e, 0.5; alpha = 0.5, beta = 0.5)] .= 1
    return y, stimuli_matrix, nothing
end

"""
    subject_selection_process(stimuli::AbstractArray{T}, target_signal::AbstractMatrix{T}) where {T<:Real}
"""
function subject_selection_process(stimuli::AbstractArray{T},
        target_signal::AbstractMatrix{T}) where {T <: Real}
    return subject_selection_process(stimuli, vec(target_signal))
end

# TODO: use Unitful to add dimensions to these values.
"""
    crop_signal!(audio::SampleBuf{T, I}; start=0, stop=1) where {T, I}

Crops an audio buffer to between `start` and `stop` in seconds.
"""
function crop_signal!(audio::AbstractSampleBuf{T, I}; start = 0, stop = 1) where {T, I}
    fs = samplerate(audio)
    audio.data = audio.data[(Int(fs * start) + 1):(Int(fs * stop)), :]
    return audio
end

# TODO: use Unitful to add dimensions to these values.
"""
    crop_signal(audio::SampleBuf{T, I}; start=0, stop=1) where {T, I}

Returns an audio buffer cropped to between `start` and `stop` in seconds.

See also [`crop_signal!`](@ref).
"""
function crop_signal(audio::AbstractSampleBuf{T, I}; start = 0, stop = 1) where {T, I}
    fs = samplerate(audio)
    return audio[(Int(fs * start) + 1):(Int(fs * stop))]
end

function DSP.stft(audio::AbstractSampleBuf{T, I}, args...; kwargs...) where {T, I}
    s = float(vec(audio.data))
    S = stft(s, args...; kwargs...)
    return S
end

"""
Crops a signal from `0:duration`, where `duration` is in seconds,
computes the short-time Fourier transform,
converts to decibels,
and then averages across STFT windows.

# Arguments
- `audio::Union{AbstractSampleBuf, Matrix}`
- `duration::Real`: duration in seconds

# Example Usage
```julia
audio = load(audio_file_path)
spect = wav2spect(audio; duration = 1.0)
```
"""
function wav2spect end

"""
    wav2spect(audio::AbstractSampleBuf{T, I}; duration = 0.5) where {T, I}
"""
function wav2spect(audio::AbstractSampleBuf{T, I}; duration = 0.5) where {T, I}
    crop_signal!(audio; start = 0, stop = duration)
    samples = length(audio)
    fs = samplerate(audio)
    S = stft(audio, samples ÷ 4, div(samples ÷ 4, 2); nfft = samples - 1, fs = fs,
        window = hamming)

    return mean(dB.(abs.(S)); dims = 2)
end

"""
    wav2spect(audio::Matrix{T}; duration = 0.5, fs = 41000.0) where T

If `audio` is a `Matrix`, try to convert to a `SampleBuf` first.
"""
function wav2spect(audio::Matrix{T}; duration = 0.5, fs = 41000.0) where {T}
    audio_ = SampleBuf(audio, fs)
    return wav2spect(audio_; duration)
end

@doc raw"""
    dB(x)

Convert from amplitude-scale to decibel-scale via

``\mathrm{dB}(x) = 10 \mathrm{log10}(x)``

# Examples
```jldoctest

julia> TinnitusReconstructor.dB.([1, 2, 100])
3-element Vector{Float64}:
  0.0
  3.010299956639812
 20.0
````

# See also
* [`invdB`](@ref)

"""
dB(x) = oftype(x / 1, 10) * log10(x)

@doc raw"""
    invdB(x)

Convert from decibel-scale to amplitude-scale via

``\mathrm{invdB}(x) = 10^{x/10}``

# Examples
```jldoctest
julia> TinnitusReconstructor.invdB.([-100, 0, 1, 2, 100])
5-element Vector{Float64}:
 1.0e-10
 1.0
 1.2589254117941673
 1.5848931924611136
 1.0e10
```

# See also
* [`dB`](@ref)
"""
invdB(x) = oftype(x / 1, 10)^(x / oftype(x / 1, 10))

"""
    rescale(X::AbstractVecOrMat{T}, min_val::Real = 0, max_val::Real = 1) where {T <: Real}

Rescales the columns of `X` to between `min_val` and `max_val`.
"""
function rescale(
        X::AbstractVecOrMat{T}, min_val::Real = 0, max_val::Real = 1) where {T <: Real}
    @assert(min_val<=max_val, "Lower bound must be less than or equal to upper bound")
    (X .- minimum(X; dims = 1)) ./ (maximum(X; dims = 1) - minimum(X; dims = 1)) .*
    (max_val - min_val) .+ min_val
end

"""
Rescales the columns of `X` to between `min_val` and `max_val` in place.
Returns the same type as the input. Supports Vector and Matrix.
"""
function rescale! end

"""
    rescale!(X::AbstractVector{T}, min_val::Real = 0, max_Val::Real = 1) where {T <: Real}
"""
function rescale!(
        X::AbstractVector{T}, min_val::Real = 0, max_val::Real = 1) where {T <: Real}
    @assert(min_val<=max_val, "Lower bound must be less than or equal to upper bound")
    X[:] = (X .- minimum(X)) ./ (maximum(X) - minimum(X)) .* (max_val - min_val) .+ min_val
end

"""
    rescale!(X::AbstractMatrix{T}) where {T}
"""
function rescale!(
        X::AbstractMatrix{T}, min_val::Real = 0, max_val::Real = 1) where {T <: Real}
    @assert(min_val<=max_val, "Lower bound must be less than or equal to upper bound")
    X[:, :] = (X .- minimum(X; dims = 1)) ./
              (maximum(X; dims = 1) - minimum(X; dims = 1)) .* (max_val - min_val) .+
              min_val
end

"""
    white_noise(Fs::T, dur::Q) where {T <: Real, Q <: Real}

Generate a white noise waveform according to the sample rate `Fs` and duration `dur`.
"""
white_noise(Fs::T, dur::Q) where {T <: Real, Q <: Real} = rand(Normal(), Int(Fs * dur))

"""
    semitones(init_freq::Real, n::Integer = 12, dir::String = "up")

Generates a `n+1` long vector of frequency values spaced by semitones starting at `init_freq`.
    `dir` can be "up" or "down" specifying if tones are ascending or descending from `init_freq`.
"""
function semitones(init_freq::Real, n::Integer = 12, dir::String = "up")
    tones = zeros(n + 1)
    tones[1] = init_freq
    for i in eachindex(tones)[2:end]
        if dir === "up"
            tones[i] = 2^(1 / 12) * tones[i - 1]
        elseif dir === "down"
            tones[i] = 2^(-1 / 12) * tones[i - 1]
        else
            error("Unknown direction: '$dir' specified")
        end
    end
    return tones
end

"""
    gen_octaves(min_freq::Real, max_freq::Real, n_in_oct::Integer = 0, spacing_type::String = "semitone")

Returns a vector of octaves between `min_freq` and `max_freq` with `n_in_oct` values between octaves.
    Spacing type can be "semitone" or "linear", indicating the distance between intra-octave values.
"""
function gen_octaves(min_freq::Real, max_freq::Real, n_in_oct::Integer = 0,
        spacing_type::String = "semitone")
    n_octs = convert(Int, floor(log2(max_freq / min_freq))) # Number of octaves between min and max
    oct_vals = min_freq * 2 .^ (0:n_octs) # Octave frequency values

    freqs = zeros(length(oct_vals) + (n_in_oct * n_octs))
    oct_marks = 1:(n_in_oct + 1):length(freqs)

    for i in 1:n_octs
        if spacing_type === "linear"
            freqs[oct_marks[i]:(oct_marks[i] + n_in_oct + 1)] .= range(
                oct_vals[i], oct_vals[i + 1], n_in_oct + 2)
        elseif spacing_type === "semitone"
            half_steps = semitones(oct_vals[i])
            inds = range(1, length(half_steps), n_in_oct + 2)

            try
                inds = convert.(Int, inds)
            catch
                error("Unable to break semitone scaling into $n_in_oct intervals inside an octave")
            end

            freqs[oct_marks[i]:(oct_marks[i] + n_in_oct + 1)] .= half_steps[inds]
        else
            error("Invalid spacing_type value. Valid options are: 'semitone', 'linear'.")
        end
    end
    return freqs
end

"""
    pure_tone(tone_freq::Real, dur::Real = 0.5, Fs::Real = 44100)

Returns a `dur`-second long pure tone waveform at `tone_freq` frequency and sample rate `Fs`.
"""
function pure_tone(tone_freq::Real, dur::Real = 0.5, Fs::Real = 44100)
    sin.(2π * tone_freq * (0:(1 / Fs):dur))
end
