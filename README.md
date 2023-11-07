# TinnitusReconstructor

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://the-lammert-lab.github.io/TinnitusReconstructor.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://the-lammert-lab.github.io/TinnitusReconstructor.jl/dev/)
[![Build Status](https://github.com/The-Lammert-Lab/TinnitusReconstructor.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/The-Lammert-Lab/TinnitusReconstructor.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/The-Lammert-Lab/TinnitusReconstructor.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/The-Lammert-Lab/TinnitusReconstructor.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

This package provides both a range of auditory stimulus generation techniques as well as 
reverse correlation reconstruction (RC) functions. 


The stimulus generation methods are intended to be used in RC experimental paradigms 
related to reconstructing the internal representations of tinnitus. 
However, the functionality is generic and may be useful in many other scenarios.

## Example usage
Create a stimgen struct using some default and some custom parameters.
```julia
using TinnitusReconstructor
stimgen = GaussianPrior(; 
        n_bins=80,
        n_bins_filled_mean=20,
        n_bins_filled_var=0.01,
    )
# GaussianPrior{Float64, Int64}(100.0, 22000.0, 0.5, 44100.0, 0.01, 20, 80)
```

Load in the target sound and convert to binned representation
```julia
using FileIO: load
import LibSndFile
audio_file = "ATA/ATA_Tinnitus_Buzzing_Tone_1sec.wav" # File path.
audio = wav2spect(load(audio_file)) # Read in file, truncate to 0.5s, convert to spectrum.
target_signal = 10 * log10.(audio) # Convert to dB

# Convert to binned representation that matches the number of stimgen bins
binned_target_signal = spect2binnedrepr(stimgen, target_signal) 
```

Generate 500 random stimuli and simulate ideal responses for each.
Then, compute the linear (`gs`) and compressed sensing (`cs`) reconstructions
and correlate the reconstruction with the binned target signal to determine reconstruction quality.
```julia
using Statistics
responses, _, stim = subject_selection_process(stimgen, target_signal, 500)
recon_linear = gs(responses, stim')
recon_cs = cs(responses, stim')
r_linear = cor(recon_linear, binned_target_signal)
r_cs = cor(recon_cs, binned_target_signal)
```

## Citation
```
@article{Hoyland2023,
	author={Hoyland, Alec and Barnett, Nelson V. and Roop, Benjamin W. and Alexandrou, Danae and Caplan, Myah and Mills, Jacob and Parrell, Benjamin and Chari, Divya A. and Lammert, Adam C.},
	journal={IEEE Open Journal of Engineering in Medicine and Biology}, 
	title={Reverse Correlation Uncovers More Complete Tinnitus Spectra}, 
	year={2023},
	volume={4},
	number={},
	pages={116-118},
	doi={10.1109/OJEMB.2023.3275051}
}
```
