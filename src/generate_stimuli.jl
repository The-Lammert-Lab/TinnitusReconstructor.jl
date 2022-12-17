# UniformPrior
function generate_stimulus(s::UniformPrior)
    # Define Frequency Bin Indices 1 through self.n_bins
    binnum, Fs, nfft, frequency_vector = freq_bins(s)
    spect = empty_spectrum(s)

    # sample from uniform distribution to get the number of bins to fill
    n_bins_to_fill = rand((s.min_bins):(s.max_bins))
    bins_to_fill = sample(1:(s.n_bins), n_bins_to_fill; replace=false)

    # Set spectrum ranges corresponding to bins to 0dB.
    [spect[binnum .== bins_to_fill[i]] .= 0 for i in 1:n_bins_to_fill]

    # Synthesize Audio
    stim = synthesize_audio(spect, nfft)

    # get the binned representation
    binned_repr = -20 * ones(s.n_bins)
    binned_repr[bins_to_fill] .= 0

    return stim, Fs, spect, binned_repr, frequency_vector
end
