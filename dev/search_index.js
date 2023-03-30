var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = TinnitusReconstructor","category":"page"},{"location":"#TinnitusReconstructor","page":"Home","title":"TinnitusReconstructor","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for TinnitusReconstructor.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [TinnitusReconstructor]","category":"page"},{"location":"#TinnitusReconstructor.unfilled_db","page":"Home","title":"TinnitusReconstructor.unfilled_db","text":"dB level of unfilled bins\n\n\n\n\n\n","category":"constant"},{"location":"#TinnitusReconstructor.Bernoulli-Tuple{}","page":"Home","title":"TinnitusReconstructor.Bernoulli","text":"Bernoulli(; kwargs...) <: BinnedStimgen\n\nConstructor for stimulus generation type in which      each tonotopic bin has a probability bin_prob     of being at 0 dB, otherwise it is at -100 dB.\n\nKeywords\n\nmin_freq::Real = 100: The minimum frequency in range from which to sample.\nmax_freq::Real = 22e3: The maximum frequency in range from which to sample.\nduration::Real = 0.5: The length of time for which stimuli are played in seconds.\nFs::Real = 44.1e3: The frequency of the stimuli in Hz.\nn_bins::Integer = 100: The number of bins into which to partition the frequency range.\nbin_prob::Real=0.3: The probability of a bin being filled.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.BinnedStimgen","page":"Home","title":"TinnitusReconstructor.BinnedStimgen","text":"BinnedStimgen <: Stimgen\n\nAbstract supertype for all binned stimulus generation.\n\n\n\n\n\n","category":"type"},{"location":"#TinnitusReconstructor.Brimijoin-Tuple{}","page":"Home","title":"TinnitusReconstructor.Brimijoin","text":"Brimijoin(; kwargs...) <: BinnedStimgen\n\nConstructor for stimulus generation type in which      each tonotopic bin is filled with an amplitude      value from an equidistant list with equal probability.\n\nKeywords\n\nmin_freq::Real = 100: The minimum frequency in range from which to sample.\nmax_freq::Real = 22e3: The maximum frequency in range from which to sample.\nduration::Real = 0.5: The length of time for which stimuli are played in seconds.\nFs::Real = 44.1e3: The frequency of the stimuli in Hz.\nn_bins::Integer = 100: The number of bins into which to partition the frequency range.\namp_min::Real = -20: The lowest dB value a bin can have.\namp_max::Real = 0: The highest dB value a bin can have.\namp_step::Int = 6: The number of evenly spaced steps between amp_min and amp_max. \n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.BrimijoinGaussianSmoothed-Tuple{}","page":"Home","title":"TinnitusReconstructor.BrimijoinGaussianSmoothed","text":"BrimijoinGaussianSmoothed(; kwargs...) <: BinnedStimgen\n\nConstructor for stimulus generation type in which      each tonotopic bin is filled by a Gaussian      with a maximum amplitude value chosen     from an equidistant list with equal probability.\n\nKeywords\n\nmin_freq::Real = 100: The minimum frequency in range from which to sample.\nmax_freq::Real = 22e3: The maximum frequency in range from which to sample.\nduration::Real = 0.5: The length of time for which stimuli are played in seconds.\nFs::Real = 44.1e3: The frequency of the stimuli in Hz.\nn_bins::Integer = 100: The number of bins into which to partition the frequency range.\namp_min::Real = -20: The lowest dB value a bin can have.\namp_max::Real = 0: The highest dB value a bin can have.\namp_step::Int = 6: The number of evenly spaced steps between amp_min and amp_max. \n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.GaussianNoise-Tuple{}","page":"Home","title":"TinnitusReconstructor.GaussianNoise","text":"GaussianNoise(; kwargs...) <: BinnedStimgen\n\nConstructor for stimulus generation type in which      each tonotopic bin is filled with amplitude chosen from a Gaussian distribution.\n\nKeywords\n\nmin_freq::Real = 100: The minimum frequency in range from which to sample.\nmax_freq::Real = 22e3: The maximum frequency in range from which to sample.\nduration::Real = 0.5: The length of time for which stimuli are played in seconds.\nFs::Real = 44.1e3: The frequency of the stimuli in Hz.\nn_bins::Integer = 100: The number of bins into which to partition the frequency range.\namplitude_mean::Real = -10: The mean of the Gaussian. \namplitude_var::Real = 3: The variance of the Gaussian. \n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.GaussianNoiseNoBins-Tuple{}","page":"Home","title":"TinnitusReconstructor.GaussianNoiseNoBins","text":"GaussianNoiseNoBins(; kwargs...) <: Stimgen\n\nConstructor for stimulus generation type in which      each frequency's amplitude is chosen according to a Gaussian distribution.\n\nKeywords\n\nmin_freq::Real = 100: The minimum frequency in range from which to sample.\nmax_freq::Real = 22e3: The maximum frequency in range from which to sample.\nduration::Real = 0.5: The length of time for which stimuli are played in seconds.\nFs::Real = 44.1e3: The frequency of the stimuli in Hz.\namplitude_mean::Real = -10: The mean of the Gaussian. \namplitude_var::Real = 3: The variance of the Gaussian. \n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.GaussianPrior-Tuple{}","page":"Home","title":"TinnitusReconstructor.GaussianPrior","text":"GaussianPrior(; kwargs...) <: BinnedStimgen\n\nConstructor for stimulus generation type in which      the number of filled bins is selected from      from a Gaussian distribution with known mean and variance parameters.\n\nKeywords\n\nmin_freq::Real = 100: The minimum frequency in range from which to sample.\nmax_freq::Real = 22e3: The maximum frequency in range from which to sample.\nduration::Real = 0.5: The length of time for which stimuli are played in seconds.\nFs::Real = 44.1e3: The frequency of the stimuli in Hz.\nn_bins::Integer = 100: The number of bins into which to partition the frequency range.\nn_bins_filled_mean::Integer = 20: The mean number of bins that may be filled on any stimuli.\nn_bins_filled_var::Real = 1: The variance of number of bins that may be filled on any stimuli.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.PowerDistribution-Tuple{}","page":"Home","title":"TinnitusReconstructor.PowerDistribution","text":"PowerDistribution(; kwargs...) <: BinnedStimgen\n\nConstructor for stimulus generation type in which      the frequencies in each bin are sampled      from a power distribution learned     from tinnitus examples.\n\nKeywords\n\nmin_freq::Real = 100: The minimum frequency in range from which to sample.\nmax_freq::Real = 22e3: The maximum frequency in range from which to sample.\nduration::Real = 0.5: The length of time for which stimuli are played in seconds.\nFs::Real = 44.1e3: The frequency of the stimuli in Hz.\nn_bins::Integer = 100: The number of bins into which to partition the frequency range.\ndistribution_filepath::AbstractString=joinpath(@__DIR__, \"distribution.csv\"): The filepath to the default power distribution from which stimuli are generated\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.Stimgen","page":"Home","title":"TinnitusReconstructor.Stimgen","text":"Abstract supertype for all stimulus generation.\n\n\n\n\n\n","category":"type"},{"location":"#TinnitusReconstructor.UniformNoise-Tuple{}","page":"Home","title":"TinnitusReconstructor.UniformNoise","text":"UniformNoise(; kwargs...) <: BinnedStimgen\n\nConstructor for stimulus generation type in which      each tonotopic bin is filled with amplitude chosen from a Uniform distribution.\n\nKeywords\n\nmin_freq::Real = 100: The minimum frequency in range from which to sample.\nmax_freq::Real = 22e3: The maximum frequency in range from which to sample.\nduration::Real = 0.5: The length of time for which stimuli are played in seconds.\nFs::Real = 44.1e3: The frequency of the stimuli in Hz.\nn_bins::Integer = 100: The number of bins into which to partition the frequency range.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.UniformNoiseNoBins-Tuple{}","page":"Home","title":"TinnitusReconstructor.UniformNoiseNoBins","text":"UniformNoiseNoBins(; kwargs...) <: Stimgen\n\nConstructor for stimulus generation type in which      each frequency is chosen from a uniform distribution on [-100, 0] dB.\n\nKeywords\n\nmin_freq::Real = 100: The minimum frequency in range from which to sample.\nmax_freq::Real = 22e3: The maximum frequency in range from which to sample.\nduration::Real = 0.5: The length of time for which stimuli are played in seconds.\nFs::Real = 44.1e3: The frequency of the stimuli in Hz.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.UniformPrior-Tuple{}","page":"Home","title":"TinnitusReconstructor.UniformPrior","text":"UniformPrior(; kwargs...) <: BinnedStimgen\n\nConstructor for stimulus generation type in which      the number of filled bins is selected from      the Uniform distribution on the interval [min_bins, max_bins].\n\nKeywords\n\nmin_freq::Real = 100: The minimum frequency in range from which to sample.\nmax_freq::Real = 22e3: The maximum frequency in range from which to sample.\nduration::Real = 0.5: The length of time for which stimuli are played in seconds.\nFs::Real = 44.1e3: The frequency of the stimuli in Hz.\nn_bins::Integer = 100: The number of bins into which to partition the frequency range.\nmin_bins::Integer = 10: The minimum number of bins that may be filled on any stimuli.\nmax_bins::Integer = 50: The maximum number of bins that may be filled on any stimuli.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.UniformPriorWeightedSampling-Tuple{}","page":"Home","title":"TinnitusReconstructor.UniformPriorWeightedSampling","text":"UniformPriorWeightedSampling(; kwargs...) <: BinnedStimgen\n\nConstructor for stimulus generation type in which      each tonotopic bin is filled from a uniform distribution on [min_bins, max_bins]     but which bins are filled is determined by a non-uniform distribution.\n\nKeywords\n\nmin_freq::Real = 100: The minimum frequency in range from which to sample.\nmax_freq::Real = 22e3: The maximum frequency in range from which to sample.\nduration::Real = 0.5: The length of time for which stimuli are played in seconds.\nFs::Real = 44.1e3: The frequency of the stimuli in Hz.\nn_bins::Integer = 100: The number of bins into which to partition the frequency range.\nmin_bins::Integer = 10: The minimum number of bins that may be filled on any stimuli.\nmax_bins::Integer = 50: The maximum number of bins that may be filled on any stimuli.\nalpha_::Real = 1: The tuning parameter that exponentiates the number of unique frequencies in each bin.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.binnedrepr2spect-Union{Tuple{T}, Tuple{BS}, Tuple{BS, AbstractArray{T}}} where {BS<:TinnitusReconstructor.BinnedStimgen, T}","page":"Home","title":"TinnitusReconstructor.binnedrepr2spect","text":"binnedrepr2spect(s::BinnedStimgen, binned_repr::AbstractArray{T}) where {BS<:BinnedStimgen,T}\n\nConvert the binned representation into a spectral representation.\n\nReturns an n_frequencies x n_trials spectral array, where n_trials = size(binned_repr, 2).\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.build_distribution-Tuple{PowerDistribution}","page":"Home","title":"TinnitusReconstructor.build_distribution","text":"build_distribution(s::PowerDistribution; save_path::AbstractString=@__DIR__)\n\nBuilds the default power distribution from ATA tinnitus sample files.     Saves the distribution as a vector in dB at save_path.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.crop_signal!-Union{Tuple{SampledSignals.AbstractSampleBuf{T, I}}, Tuple{I}, Tuple{T}} where {T, I}","page":"Home","title":"TinnitusReconstructor.crop_signal!","text":"crop_signal!(audio::SampleBuf{T, I}; start=0, stop=1) where {T, I}\n\nCrops an audio buffer to between start and stop in seconds. TODO: use Unitful to add dimensions to these values.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.crop_signal-Union{Tuple{SampledSignals.AbstractSampleBuf{T, I}}, Tuple{I}, Tuple{T}} where {T, I}","page":"Home","title":"TinnitusReconstructor.crop_signal","text":"crop_signal(audio::SampleBuf{T, I}; start=0, stop=1) where {T, I}\n\nReturns an audio buffer cropped to between start and stop in seconds. TODO: use Unitful to add dimensions to these values.\n\nSee also crop_signal!.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.cs-Union{Tuple{T}, Tuple{Any, AbstractArray{T}}, Tuple{Any, AbstractArray{T}, Integer}} where T<:Real","page":"Home","title":"TinnitusReconstructor.cs","text":"cs(responses, Φ, Γ::Integer=32)\n\nOne-bit compressed sensing reverse correlation with basis.\n\nReferences\n\nPlan, Y. and Vershynin, R., 2012. Robust 1-bit compressed sensing and sparse logistic regression: A convex programming approach. IEEE Transactions on Information Theory, 59(1), pp.482-494.\nZhang, L., Yi, J. and Jin, R., 2014, June. Efficient algorithms for robust one-bit compressive sensing. In International Conference on Machine Learning (pp. 820-828). PMLR.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.empty_spectrum-Tuple{BS} where BS<:TinnitusReconstructor.BinnedStimgen","page":"Home","title":"TinnitusReconstructor.empty_spectrum","text":"empty_spectrum(s::BS) where {BS<:BinnedStimgen}\n\nGenerate an nfft x 1 vector of Ints, where all values are -100. \n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.freq_bins-Tuple{BS} where BS<:TinnitusReconstructor.BinnedStimgen","page":"Home","title":"TinnitusReconstructor.freq_bins","text":"freq_bins(s::BS) where {BS<:BinnedStimgen}\n\nGenerates a vector indicating which frequencies belong to the same bin,     following a tonotopic map of audible frequency perception.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.fs-Tuple{SG} where SG<:TinnitusReconstructor.Stimgen","page":"Home","title":"TinnitusReconstructor.fs","text":"fs(s::SG) where {SG<:Stimgen}\n\nReturn the number of samples per second.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.generate_stimuli_matrix-Union{Tuple{I}, Tuple{BS}, Tuple{BS, I}} where {BS<:TinnitusReconstructor.BinnedStimgen, I<:Integer}","page":"Home","title":"TinnitusReconstructor.generate_stimuli_matrix","text":"generate_stimuli_matrix(s::BS, n_trials::I) where {BS<:BinnedStimgen, I<:Integer}\n\nGenerate n_trials of stimuli based on specifications in the stimgen type.\n\nReturns stimuli_matrix, Fs, spect_matrix, binned_repr_matrix. \n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.generate_stimuli_matrix-Union{Tuple{I}, Tuple{SG}, Tuple{SG, I}} where {SG<:TinnitusReconstructor.Stimgen, I<:Integer}","page":"Home","title":"TinnitusReconstructor.generate_stimuli_matrix","text":"generate_stimuli_matrix(s::SG, n_trials::I) where {SG<:Stimgen, I<:Integer}\n\nGenerate n_trials of stimuli based on specifications in the stimgen type.\n\nReturns stimuli_matrix, Fs, spect_matrix, binned_repr_matrix.      binned_repr_matrix = nothing if s >: BinnedStimgen.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.generate_stimulus","page":"Home","title":"TinnitusReconstructor.generate_stimulus","text":"generate_stimulus(s::Stimgen)\n\nGenerate one stimulus sound.\n\nReturns waveform, sample rate, spectral representation, binned representation, and a frequency vector. Methods are specialized for each concrete subtype of Stimgen.\n\n\n\n\n\n","category":"function"},{"location":"#TinnitusReconstructor.gs-Tuple{Any, Any}","page":"Home","title":"TinnitusReconstructor.gs","text":"gs(responses, Φ)\n\nLinear reverse correlation.\n\nReferences\n\nGosselin, F. and Schyns, P.G., 2003. Superstitious perceptions reveal properties of internal representations. Psychological science, 14(5), pp.505-509.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.nsamples-Tuple{SG} where SG<:TinnitusReconstructor.Stimgen","page":"Home","title":"TinnitusReconstructor.nsamples","text":"nsamples(s::SG) where {SG<:Stimgen}\n\nReturn the number of samples as an Integer. This means that the product fs(s) * s.duration must be an Integer or an InexactError will be thrown.\n\nExamples\n\n```jldoctest julia> s = UniformPrior(;Fs=44.1e3, duration=0.5); nsamples(s) 22050\n\njulia> s = UniformPrior(;Fs=44.1e3, duration=0.7); nsamples(s) ERROR: InexactError: Int64(30869.999999999996)\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.play_scaled_audio-Tuple{Any, Any}","page":"Home","title":"TinnitusReconstructor.play_scaled_audio","text":"play_scaled_audio(x, Fs)\n\nScales audio signal from -1 to 1 then plays it. Adapted from MATLAB's soundsc().\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.soft_threshold-Tuple{Any, Any}","page":"Home","title":"TinnitusReconstructor.soft_threshold","text":"soft_threshold(α, γ)\n\nSoft thresholding operator for use in compressed sensing.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.spect2binnedrepr-Union{Tuple{T}, Tuple{BS}, Tuple{BS, AbstractVecOrMat{T}}} where {BS<:TinnitusReconstructor.BinnedStimgen, T}","page":"Home","title":"TinnitusReconstructor.spect2binnedrepr","text":"spect2binnedrepr(s::BinnedStimgen, spect::AbstractVecOrMat{T}) where {BS<:BinnedStimgen,T}\n\nConvert a spectral representation into a binned representation.\n\nReturns an n_trials x n_bins array containing the amplitude of the spectrum in each frequency bin,     where n_trials = size(binned_repr, 2). @doc \n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.subject_selection_process-Tuple{AbstractArray, AbstractMatrix}","page":"Home","title":"TinnitusReconstructor.subject_selection_process","text":"subject_selection_process(stimuli::AbstractArray{T}, target_signal::AbstractMatrix{T}) where {T<:Real}\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.subject_selection_process-Tuple{AbstractVecOrMat, AbstractVector}","page":"Home","title":"TinnitusReconstructor.subject_selection_process","text":"subject_selection_process(stimuli_matrix::AbstractVecOrMat{T}, target_signal::AbstractVector{T}) where {T<:Real}\n\nPerform the synthetic subject decision process, given a matrix of precomputed stimuli stimuli_matrix and a target_signal. The stimuli_matrix is of size m x n where m is the number of trials and n is the number of samples in the signal. The target_signal is a flat vector of size n or an n x 1 matrix. Return the n-dimensional response vector y as well as the stimuli_matrix as well as nothing for the binned representation.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.subject_selection_process-Union{Tuple{I}, Tuple{T}, Tuple{SG}, Tuple{SG, AbstractMatrix{T}, I}} where {SG<:TinnitusReconstructor.Stimgen, T<:Real, I<:Integer}","page":"Home","title":"TinnitusReconstructor.subject_selection_process","text":"subject_selection_process(s::SG, target_signal::AbstractMatrix{T}, n_trials::I) where {SG<:Stimgen,T<:Real,I<:Integer}\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.subject_selection_process-Union{Tuple{I}, Tuple{T}, Tuple{SG}, Tuple{SG, AbstractVector{T}, I}} where {SG<:TinnitusReconstructor.Stimgen, T<:Real, I<:Integer}","page":"Home","title":"TinnitusReconstructor.subject_selection_process","text":"subject_selection_process(s::SG, target_signal::AbstractVector{T}, n_trials::I) where {SG<:Stimgen,T<:Real,I<:Integer}\n\nPerform the synthetic subject decision process, generating the stimuli on-the-fly using the stimulus generation method s.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.synthesize_audio-Tuple{Any, Any}","page":"Home","title":"TinnitusReconstructor.synthesize_audio","text":"synthesize_audio(X, nfft)\n\nSynthesize audio from spectrum, X\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.wav2spect-Tuple{String}","page":"Home","title":"TinnitusReconstructor.wav2spect","text":"wav2spect(audio_file::String; duration=0.5)\n\nLoad an audio file, crop it to duration,     then compute and return the short time Fourier transform.\n\n\n\n\n\n","category":"method"},{"location":"#TinnitusReconstructor.zhangpassivegamma-Union{Tuple{T}, Tuple{AbstractArray{T}, Any, Integer}} where T<:Real","page":"Home","title":"TinnitusReconstructor.zhangpassivegamma","text":"zhangpassivegamma(Φ::AbstractArray{T}, y, Γ::Integer) where {T<:Real}\n\nPassive algorithm for 1-bit compressed sensing with no basis.\n\nReferences\n\nZhang, L., Yi, J. and Jin, R., 2014, June. Efficient algorithms for robust one-bit compressive sensing. In International Conference on Machine Learning (pp. 820-828). PMLR.\n\n\n\n\n\n","category":"method"}]
}