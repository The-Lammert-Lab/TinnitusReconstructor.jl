# UniformPrior
function generate_stimulus(s::UniformPrior)

    # Define Frequency Bin Indices 1 through self.n_bins
    binnum, Fs, nfft, frequency_vector = freq_bins(s)
    spect = empty_spectrum(s)

    # Generate Random Freq Spec in dB Acccording to Frequency Bin Index

    # master list of frequency bins unfilled
    frequency_bin_list = collect(1:(s.n_bins))

    # sample from uniform distribution to get the number of bins to fill
    n_bins_to_fill = rand((s.min_bins):(s.max_bins))

    filled_bins = zeros(Int64, n_bins_to_fill, 1)
    bin_to_fill = zero(Int64)

    # fill the bins
    for i in 1:n_bins_to_fill
        # Select a bin at random from the list
        random_bin_index = rand(1:(length(frequency_bin_list)))
        bin_to_fill = frequency_bin_list[random_bin_index]
        filled_bins[i] = bin_to_fill
        # fill that bin
        spect[binnum .== bin_to_fill] .= 0
        # remove that bin from the master list
        deleteat!(frequency_bin_list, frequency_bin_list .== bin_to_fill)
    end

    # Synthesize Audio
    stim = synthesize_audio(spect, nfft)

    # get the binned representation
    binned_repr = -20 * ones(s.n_bins, 1)
    binned_repr[filled_bins] .= 0

    return stim, Fs, spect, binned_repr, frequency_vector
end
