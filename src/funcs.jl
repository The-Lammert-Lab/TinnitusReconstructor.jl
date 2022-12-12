import DSP: stft

hz2mels(f) = 2595 * log10(1 + (f / 700))

mels2hz(m) = 700 * (10^(m / 2595) - 1)

"""
    play_scaled_audio(x, Fs) 

Scales audio signal from -1 to 1 then plays it. Adapted from MATLAB's soundsc.
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

function P(alpha, gamma)
    beta = zeros(length(alpha), 1)

    ind = @. abs(alpha) > gamma

    return @. beta[ind] = sign(alpha[ind]) * (abs(alpha[ind]) - gamma)
end

function zhangpassivegamma(Phi, y, h)
    m = size(Phi, 1)
    n = size(Phi, 2)

    a = (1 / m) .* (Phi'y)

    val = sort(abs.(a); rev=true)
    gamma = val[h + 1]

    if norm(a, Inf) <= gamma
        return zeros(n, 1)
    else
        return (1 / norm(P(a, gamma), 2)) * P(a, gamma)
    end
end

"""
    gs(responses, Phi)

Linear reverse correlation.
"""
gs(responses, Phi) = (1 / size(Phi, 2)) * Phi'responses

"""
    cs(responses, Phi, Gamma::Integer=32)

Compressed sensing reverse correlation.
"""
function cs(responses, Phi, Gamma::Integer=32)
    n_samples = length(responses)
    len_signal = size(Phi, 2)

    Theta = zeros(n_samples, len_signal)
    for i in 1:len_signal
        ek = zeros(Int, len_signal, 1)
        ek[i] = 1
        Psi = idct(ek)
        Theta[:, 1] .= Phi * Psi
    end

    s = zhangpassivegamma(Theta, responses, Gamma)

    x = zeros(len_signal, 1)
    for i in 1:len_signal
        ek = zeros(Int, len_signal, 1)
        ek[i] = 1
        Psi = idct(ek)
        x .+= Psi * s[i]
    end

    return x
end

cs_no_basis(responses, Phi, Gamma=32) = zhangpassivegamma(Phi, responses, Gamma)

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
