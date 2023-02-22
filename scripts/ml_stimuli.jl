"""
This script trains a neural network
whose weights are a measurement matrix.
It then uses those weights as stimuli
for a simulated tinnitus reconstruction experiment
using the synthetic subject.
"""

push!(LOAD_PATH, "..")

using Flux
using TinnitusReconstructor
