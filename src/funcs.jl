import DSP: stft

hz2mels(f) = 2595 * log10(1 + (f / 700))

mels2hz(m) = 700 * (10^(m / 2595) - 1)

"""
    play_scaled_audio(x, Fs) 

Scales audio signal from -1 to 1 then plays it. Adapted from MATLAB's soundsc().
"""
function play_scaled_audio(x, Fs)

    # Translated MATLAB
    xmax = @. $maximum(abs, x[!isinf(x)])

    slim = [-xmax, xmax]

    dx = diff(slim)
    if iszero(dx)
        # Protect against divide-by-zero errors:
        x = zeros(size(x))
    else
        x = @. (x - slim[1]) / dx * 2 - 1
    end

    # This is using PortAudio, SampledSignals
    PortAudioStream(0, 2; samplerate=Fs) do stream
        write(stream, x)
    end

    return nothing
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
function zhangpassivegamma(Φ::AbstractArray{T}, y, Γ::Integer) where {T<:Real}
    m = size(Φ, 1)
    n = size(Φ, 2)

    α = (1 / m) * (Φ'y)

    val = sort(abs.(α); rev=true)
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
function cs(responses, Φ::AbstractArray{T}, Γ::Integer=32) where {T<:Real}
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

cs_no_basis(Φ, responses, Γ=32) = zhangpassivegamma(Φ, responses, Γ)

"""
    subject_selection_process(s::Stimgen, target_signal::AbstractVector{T}) where {T<:Real}
    subject_selection_process(s::Stimgen, target_signal::AbstractMatrix{T}) where {T<:Real}
    subject_selection_process(stimuli::AbstractArray{T}, target_signal::AbstractVector{T}) where {T<:Real}
    subject_selection_process(stimuli::AbstractArray{T}, target_signal::AbstractMatrix{T}) where {T<:Real}

Idealized model of a subject performing the task.

Specify a `Stimgen` type from which to generate stimuli or input a stimuli matrix.
Return an `n_samples x 1` vector of `-1` for "no" and `1` for "yes".
"""
function subject_selection_process end

function subject_selection_process(
    stimuli::AbstractArray{T}, target_signal::AbstractVector{T}
) where {T<:Real}
    @assert(
        !isempty(stimuli),
        "Stimuli must be explicitly passed or generated via
`subject_selection_process(s::Stimgen, target_signal::AbstractVector{T}) where {T<:Real}`"
    )

    # Ideal selection
    e = stimuli * target_signal
    y = -ones(Int, size(e))
    y[e .>= quantile(e, 0.5; alpha=0.5, beta=0.5)] .= 1

    return y, stimuli
end

# Convert target_signal to a Vector if passed as an Array.
function subject_selection_process(
    stimuli::AbstractArray{T}, target_signal::AbstractMatrix{T}
) where {T<:Real}
    @assert size(target_signal, 2) == 1 "Target signal must be a Vector or single-column Matrix."
    return subject_selection_process(stimuli, vec(target_signal))
end

"""
    crop_signal!(audio::SampleBuf{T, I}; start=0, stop=1) where {T, I}

Crops an audio buffer to between `start` and `stop` in seconds.
TODO: use Unitful to add dimensions to these values.
"""
function crop_signal!(audio::AbstractSampleBuf{T,I}; start=0, stop=1) where {T,I}
    fs = samplerate(audio)
    audio.data = audio.data[(Int(fs * start) + 1):(Int(fs * stop)), :]
    return audio
end

"""
    crop_signal(audio::SampleBuf{T, I}; start=0, stop=1) where {T, I}

Returns an audio buffer cropped to between `start` and `stop` in seconds.
TODO: use Unitful to add dimensions to these values.

See also [`crop_signal!`](@ref).
"""
function crop_signal(audio::AbstractSampleBuf{T,I}; start=0, stop=1) where {T,I}
    fs = samplerate(audio)
    return audio[(Int(fs * start) + 1):(Int(fs * stop))]
end

function DSP.stft(audio::AbstractSampleBuf{T,I}, args...; kwargs...) where {T,I}
    s = float(vec(audio.data))
    S = stft(s, args...; kwargs...)
    return S
end

"""
    wav2spect(audio_file::String; duration=0.5)

Load an audio file, crop it to `duration`,
    then compute and return the Welch power spectral density estimate.
"""
function wav2spect(audio_file::String; duration=0.5)
    audio = load(audio_file)
    crop_signal!(audio; start=0, stop=duration)

    samples = length(audio)
    fs = samplerate(audio)

    S = stft(
        audio, samples ÷ 4, div(samples ÷ 4, 2); nfft=samples - 1, fs=fs, window=hamming
    )

    return mean(abs.(S); dims=2)
end
