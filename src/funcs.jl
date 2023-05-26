import DSP: stft

hz2mels(f) = 2595 * log10(1 + (f / 700))

mels2hz(m) = 700 * (10^(m / 2595) - 1)

"""
    synthesize_audio(X, nfft)

Synthesize audio from spectrum `X`
"""
function synthesize_audio(X, nfft)
    phase = 2π * (rand(nfft ÷ 2) .- 0.5) # Assign random phase to freq spec
    s = @.. (10^(X / 10)) * cis(phase) # Convert dB to amplitudes
    ss = vcat(1, s)
    return irfft(ss, 2 * length(ss) - 1) #transform from freq to time domain
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
    subject_selection_process(stimuli_matrix::AbstractVecOrMat{T}, target_signal::AbstractVector{T}) where {T<:Real}

Perform the synthetic subject decision process, given a matrix of precomputed stimuli `stimuli_matrix`
and a `target_signal`.
The `stimuli_matrix` is of size `m x n` where `m` is the number of trials and `n` is the number of samples in the signal.
The `target_signal` is a flat vector of size `n` or an `n x 1` matrix.
Return the `n`-dimensional response vector `y` as well as the `stimuli_matrix`
as well as `nothing` for the binned representation.
"""
function subject_selection_process(
    stimuli_matrix::AbstractVecOrMat{T}, target_signal::AbstractVector{T}
) where {T<:Real}
    e = stimuli_matrix'target_signal
    y = -ones(Int, size(e))
    y[e .>= quantile(e, 0.5; alpha=0.5, beta=0.5)] .= 1
    return y, stimuli_matrix, nothing
end

"""
    subject_selection_process(stimuli::AbstractArray{T}, target_signal::AbstractMatrix{T}) where {T<:Real}
"""
function subject_selection_process(
    stimuli::AbstractArray{T}, target_signal::AbstractMatrix{T}
) where {T<:Real}
    return subject_selection_process(stimuli, vec(target_signal))
end

# TODO: use Unitful to add dimensions to these values.
"""
    crop_signal!(audio::SampleBuf{T, I}; start=0, stop=1) where {T, I}

Crops an audio buffer to between `start` and `stop` in seconds.
"""
function crop_signal!(audio::AbstractSampleBuf{T,I}; start=0, stop=1) where {T,I}
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
    then compute and return the short time Fourier transform.
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

raw"""
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

raw"""
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
