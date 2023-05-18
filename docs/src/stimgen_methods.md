```@meta
CurrentModule = TinnitusReconstructor
```

# `Stimgen Methods`

## `All stimgen`

```@docs
TinnitusReconstructor.generate_stimulus
TinnitusReconstructor.fs
TinnitusReconstructor.nsamples
TinnitusReconstructor.subject_selection_process(s::SG, target_signal::AbstractVector{T}, n_trials::I) where {SG<:Stimgen,T<:Real,I<:Integer}
TinnitusReconstructor.subject_selection_process(s::SG, target_signal::AbstractMatrix{T}, n_trials::I) where {SG<:Stimgen,T<:Real,I<:Integer}
TinnitusReconstructor.generate_stimuli_matrix(s::SG, n_trials::I) where {SG<:Stimgen, I<:Integer}

```

## `Binned only`

```@docs
TinnitusReconstructor.generate_stimuli_matrix(s::BS, n_trials::I) where {BS<:BinnedStimgen, I<:Integer}
TinnitusReconstructor.freq_bins
TinnitusReconstructor.empty_spectrum
TinnitusReconstructor.spect2binnedrepr
TinnitusReconstructor.binnedrepr2spect
TinnitusReconstructor.build_distribution
```
