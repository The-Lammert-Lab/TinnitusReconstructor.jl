import DSP: stft

hz2mels(f) = 2595 * log10(1 + (f / 700))

mels2hz(m) = 700 * (10^(m / 2595) - 1)

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

"""
dB(x) = 10log10(x)

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
* [`db⁻¹`](@ref)
"""
invdB(x) = 10^(x/10)

@doc raw"""
    db⁻¹(x)

Synonym for [`invdB`](@ref).

# See also
* [`dB`](@ref)
* [`db⁻¹`](@ref)
"""
db⁻¹(x) = @inline invdB(x)

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

# """
#     subject_selection_process(s::SG, target_signal::AbstractVector{T}, n_trials::I) where {SG<:Stimgen, T<:Real, I<:Integer}
#     subject_selection_process(s::SG, target_signal::AbstractMatrix{T}, n_trials::I) where {SG<:Stimgen, T<:Real, I<:Integer}
#     subject_selection_process(stimuli::AbstractArray{T}, target_signal::AbstractVector{T}) where {T<:Real}
#     subject_selection_process(stimuli::AbstractArray{T}, target_signal::AbstractMatrix{T}) where {T<:Real}

# Idealized model of a subject performing the task.

# Specify a `Stimgen` type from which to generate stimuli or input a stimuli matrix.
# If `target_signal` is a matrix, it must be one dimensional because that method simply applies `vec(target_signal)`.
# Return an `n_trials x 1` or `size(stimuli, 1) x 1` vector of `-1` for "no" and `1` for "yes".
# """
# function subject_selection_process end

@doc """
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

@doc """
    subject_selection_process(stimuli::AbstractArray{T}, target_signal::AbstractMatrix{T}) where {T<:Real}
"""
function subject_selection_process(
    stimuli::AbstractArray{T}, target_signal::AbstractMatrix{T}
) where {T<:Real}
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

"""
    mmd(x, y, σ=1)

Compute the maximum mean discrepancy (MMD)
between `x` and `y` using a Gaussian kernel.

# Examples
TODO
"""
function mmd(x, y, σ=1)
    M = length(x)
    N = length(y)

    mmd = 0

    running_total = 0
    for i in 1:M, j in 1:M
        running_total += gaussian_kernel(x[i], x[j])
    end
    mmd += (running_total / M^2)

    running_total = 0
    for i in 1:M, j in 1:N
        running_total += gaussian_kernel(x[i], y[j])
    end
    mmd -= (2 / (M * N) * running_total)

    running_total = 0
    for i in 1:N, j in 1:N
        running_total += gaussian_kernel(y[i], y[j])
    end
    mmd += (running_total / N^2)

    return mmd
end

@doc raw"""
    gaussian_kernel(x, y; σ=1)

Compute the gaussian kernel for `x` and `y`.
This is the function

``k_\sigma : \mathbb{R}^{2m} \times \mathbb{R}^{2m} \rightarrow \mathbb{R}, (x, y) \mapsto k_\sigma (x, y) = \exp \left ( - \frac{1}{2\sigma^2} ||x-y||^2 \right )`` 

# Examples
```jldoctest
julia> TinnitusReconstructor.gaussian_kernel(1, 1)
1.0
```
"""
function gaussian_kernel(x, y; σ=1)
    return @. exp(-1 / (2 * σ^2) * abs(x - y)^2)
end

@doc raw"""
    phase_to_mm(Φ)

Convert a matrix of phases `Φ` to a measurement matrix via
``\frac{1}{\sqrt{m}} \exp(i \Phi)``.
"""
phase_to_mm(Φ) = 1 / sqrt(size(Φ, 1)) * cis(Φ)

@doc raw"""
    stk(z)

Stack real and imaginary parts of a complex vector `z`
in a real vector `stk(z)`:

``\mathrm{stk} : \mathbb{C}^m \rightarrow \mathbb{R}^{2m}, z \mapsto \mathrm{stk}(z) = \left[\mathcal{R}(z)^{\mathrm{T}}, \mathcal{I}(z)^{\mathrm{T}} \right]^{\mathrm{T}}``
"""
function stk(z)
    return vcat(vec(real(z)'), vec(imag(z)'))
end
