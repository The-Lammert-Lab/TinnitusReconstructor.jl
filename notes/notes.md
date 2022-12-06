# Development notes

## Week of 11/21/22

### 3-autoformat-code

- Autoformatting is difficult to do because 
Julia is saved to different locations depending 
on the operating system. As far as I can tell, it can't be 
run directly from the terminal without some user-specific setup.

UPDATE: from [here](https://stackoverflow.com/questions/65243488/portable-shebang-line-for-julia-with-command-line-options)

```
#!/bin/bash
#=
exec julia -O0 --compile=min "${BASH_SOURCE[0]}" "$@"
=#
```

- `FormatCheck.yml` is responsible for checking the formatting
but does not actually format the code.

### 1-update-cicd-to-test-on-julia-18

- TagBot requires an SSH Deploy Key in order to run
other actions (like documentation). See [here](https://github.com/JuliaRegistries/TagBot#ssh-deploy-keys). 
This is a repository-owner task, I believe.

    - Permissions section in `TagBot.yml` might not be necessary. 
    It's not used in SciML's TagBot! I kept it in because that's what they suggest.

    - I removed the default lookback in workflow_dispatch
    because vscode was giving an error. Couldn't easily sort out
    the error and I don't think we'll be regularly running this action.


### 2-stimulus-generation

- How to print the actual kwarg name in error message?

    - **Assert macro**

- Is there a good way to parameterize the generate_stimulus functoin? 
Since each version is different depending on the method, it seems like a concrete type input needs to be written for each one.

- Will we need all of the stimgen strategies as their own types?
    - Have the strategy determined by a field?
    - Having trouble coming up with a more efficient/Julian way of doing it.

- Duck typing for hz2mels, etc., does not affect performance.

- Will `Fs` and `nfft` always be ints? 
Wrote getter function to always return Int64 because of trouble with zeros(), range(), in freq_bins. 
    - If yes, is there any benefit in specifying type for things like `synthesize_audio(X, nfft)`, 
    as in `synthesize_audio(X, nfft::Int64)`?

- Why the while codition in loop for UniformPrior `generate_stimulus()`?

- Where to put `using` in file? Always at the top of file? In main module file?

- Is `get_Fs()` necessary? `get_nfft()` can be written easily without it.
Plus, getter functions are more useful for things that are exported to discourage direct field acces, I think.
But that in itself requires blocking of other fields.

- Naming convention for Julia usually doesn't have "get", but then names get crossed up when assigning to variables. 

- `ifft()` from the FFTW library does not return a real valued array for complex input (`ss` in `synthesize_audio(X, nfft)`). 
I imagine this is/will be an issue, but am unsure of the fix since conversion from 
Complex{Float64} &rarr; Array{Float64} runs into an InexactError. 
Currently I just take the real part as a workaround.

- It seems like the "include" order matters.

- Getting Warning 
```
libportaudio: Output underflowed
```
on PortAudioStream.

---

## Notes from meeting with Alec 11/29/22

Use @.

Use [@assert](https://docs.julialang.org/en/v1/base/base/#Base.@assert)

Split generate_stimuli_matrix function into binned and unbinned.

[Styleguide](https://docs.julialang.org/en/v1/manual/style-guide/)

Check out LibSndFile

Build synthetic subject and do reconstruction with it using cs and lr.

## Week of 11/28/22

- LibSndFile is for reading and writing audio, PortAudio is still good for playing.

- Improve function names?
    - `P()` in `zhangpassivegamma()`

- Check `cs_no_basis()` implementation.

- Instead of importing a whole package, extending `quantile()` to get
equivalent of MATLAB's `prctile()` result. percentile(x, p) = quantile(x, p / 100).
See [here](https://github.com/JuliaStats/StatsBase.jl/blob/88b481809cf3a4b4e381be37f4372122c2d7c361/src/scalarstats.jl#L184-L188)
This brings up a problem with the simulated responses, though. 
Julia has many options for quantile (see [here](https://docs.julialang.org/en/v1/stdlib/Statistics/#Statistics.quantile)). MATLAB defaults to alpha = 0.5, beta = 0.5.
Julia, R, NumPy, all default to alpha = 1, beta = 1. What to use??

- Why is subject_selection_process a class method as well as a standalone function?

- Example on LibSndFile use `import`. Is that better? Worse?

- `wav2spect` what if index isn't int? Added round(Int, x).