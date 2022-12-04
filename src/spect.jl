using FileIO
using SampledSignals
using DSP
import DSP: spectrogram
using Statistics

"""
    DSP.spectrogram(audio::AbstractSampleBuf{T, I}; kwargs...) where {T, I}

Computes a spectrogram from an audio buffer.
"""
function DSP.spectrogram(audio::AbstractSampleBuf{T,I}; kwargs...) where {T,I}
    s = float(vec(audio.data))
    S = spectrogram(s, kwargs...)
    return S
end

"""
    crop_signal!(audio::SampleBuf{T, I}; start=0, stop=1) where {T, I}

Crops an audio buffer to between `start` and `stop` in seconds.
TODO: use Unitful to add dimensions to these values.
"""
function crop_signal(audio::SampleBuf{T,I}; start=0, stop=1) where {T,I}
    fs = samplerate(audio)
    return audio[(Int(fs * start) + 1):(Int(fs * stop))]
end

"""
    wav2spect(audio_file::String; duration=0.5)

Load an audio file, then crop it to `duration`
and finally compute and return the spectrogram.
"""
function wav2spect(audio_file::String; duration=0.5)
    audio = load(audio_file)
    audio = crop_signal(audio; start=0, stop=duration)
    S = spectrogram(audio)
    return S
end