```@meta
CurrentModule = TinnitusReconstructor
```

# `Utility Functions`

## `Prediction related functions`
```@docs
subject_selection_process(stimuli_matrix::AbstractVecOrMat{T}, target_signal::AbstractVector{T}) where {T<:Real}
```

## `Reconstruction related functions`
```@docs
TinnitusReconstructor.gs
TinnitusReconstructor.cs
TinnitusReconstructor.zhangpassivegamma
TinnitusReconstructor.soft_threshold
```

## `Audio manipulation`
```@docs
TinnitusReconstructor.synthesize_audio
TinnitusReconstructor.crop_signal!
TinnitusReconstructor.crop_signal
TinnitusReconstructor.wav2spect
TinnitusReconstructor.dB
TinnitusReconstructor.invdB
```
