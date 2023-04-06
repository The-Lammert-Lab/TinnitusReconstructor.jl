using TinnitusReconstructor

const stimgen = BrimijoinGaussianSmoothed()
@time generate_stimuli_matrix(stimgen, 200)
@time generate_stimuli_matrix(stimgen, 200)