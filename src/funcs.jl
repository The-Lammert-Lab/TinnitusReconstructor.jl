# Helper functions. Like tinnitus-project utils/

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

gs(responses, Phi) = (1 / size(Phi, 2)) * Phi'responses

function cs(responses, Phi, Gamma=32)
    n_samples = length(responses)
    len_signal = size(Phi, 2)

    Theta = zeros(n_samples, len_signal)
    for i in 1:len_signal
        ek = zeros(Int64, len_signal, 1)
        ek[i] = 1
        Psi = idct(ek)
        Theta[:, 1] .= Phi * Psi
    end

    s = zhangpassivegamma(Theta, responses, Gamma)

    x = zeros(len_signal, 1)
    for i in 1:len_signal
        ek = zeros(Int64, len_signal, 1)
        ek[i] = 1
        Psi = idct(ek)
        x .+= Psi * s[i]
    end

    return x
end

cs_no_basis(responses, Phi, Gamma=32) = zhangpassivegamma(Phi, responses, Gamma)

function subject_selection_process(
    target_signal::AbstractVector{T}, stimuli::AbstractArray{T}, n_samples=nothing
) where {T<:Real}
    target_signal = vec(target_signal)
    isempty(stimuli) ? X = round.(rand(n_samples, length(target_signal))) : X = stimuli

    # Ideal selection
    e = X * target_signal
    y = -ones(Int64, size(e))
    y[e .>= quantile(e, 0.5; alpha=0.5, beta=0.5)] .= 1

    return y, X
end
function subject_selection_process(
    target_signal::AbstractArray{T}, stimuli::AbstractArray{T}, n_samples=nothing
) where {T<:Real}
    return subject_selection_process(vec(target_signal), stimuli, n_samples)
end

function wav2spect(audio_file::String, duration=0.5)
    audio = load(audio_file)
    samples = length(audio)
    fs = samplerate(audio)

    if mod(samples, 2) != 0
        audio = audio[1:(end - 1)]
    end

    audio = audio[1:round(Int, fs * duration)]

    S = spectrogram(
        audio,
        samples ÷ 8,
        div(samples ÷ 8, 2);
        nfft=(samples - 1) ÷ 2,
        window=hamming,
        fs=fs,
    )

    return mean(abs.(power(S)); dims=2), freq(S)
end
