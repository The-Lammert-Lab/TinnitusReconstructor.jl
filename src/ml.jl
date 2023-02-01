
using Flux
using Flux: create_bias
using Functors
import Base: show
using LinearAlgebra

"""
    mmd(x, y, σ=1)

Compute the maximum mean discrepancy (MMD)
between `x` and `y` using a Gaussian kernel.

# Examples
TODO
"""
function mmd(x, y, σ=1)
    M = length(x)
    N = length(y)

    mmd = 0

    running_total = 0
    for i in 1:M, j in 1:M
        running_total += gaussian_kernel(x[i], x[j])
    end
    mmd += (running_total / M^2)

    running_total = 0
    for i in 1:M, j in 1:N
        running_total += gaussian_kernel(x[i], y[j])
    end
    mmd -= (2 / (M * N) * running_total)

    running_total = 0
    for i in 1:N, j in 1:N
        running_total += gaussian_kernel(y[i], y[j])
    end
    mmd += (running_total / N^2)

    return mmd
end

@doc raw"""
    gaussian_kernel(x, y; σ=1)

Compute the gaussian kernel for `x` and `y`.
This is the function

``k_\sigma : \mathbb{R}^{2m} \times \mathbb{R}^{2m} \rightarrow \mathbb{R}, (x, y) \mapsto k_\sigma (x, y) = \exp \left ( - \frac{1}{2\sigma^2} ||x-y||^2 \right )`` 

# Examples
```jldoctest
julia> TinnitusReconstructor.gaussian_kernel(1, 1)
1.0
```
"""
function gaussian_kernel(x, y; σ=1)
    return @. exp(-1 / (2 * σ^2) * abs(x - y)^2)
end

@doc raw"""
    phase_to_mm(Φ)

Convert a matrix of phases `Φ` to a measurement matrix via
``\frac{1}{\sqrt{m}} \exp(i \Phi)``.
"""
phase_to_mm(Φ) = 1 / sqrt(size(Φ, 1)) * cis(Φ)

@doc raw"""
    stk(z)

Stack real and imaginary parts of a complex vector `z`
in a real vector `stk(z)`:

``\mathrm{stk} : \mathbb{C}^m \rightarrow \mathbb{R}^{2m}, z \mapsto \mathrm{stk}(z) = \left[\mathcal{R}(z)^{\mathrm{T}}, \mathcal{I}(z)^{\mathrm{T}} \right]^{\mathrm{T}}``
"""
function stk(z)
    return vcat(vec(real(z)'), vec(imag(z)'))
end

@doc raw"""
    TransformedDense(in => out, σ=identity, σ2=identity; bias=true, init=glorot_uniform)
    TransformedDense(W::AbstractMatrix, [bias, σ, σ2])

    Create a traditional fully connected layer, whose forward pass is given by:
    y = σ.(σ2(W) * x .+ bias)
The input `x` should be a vector of length `in`, or batch of vectors represented
as an `in × N` matrix, or any array with `size(x,1) == in`.
The out `y` will be a vector  of length `out`, or a batch with
`size(y) == (out, size(x)[2:end]...)`
Keyword `bias=false` will switch off trainable bias for the layer.
The initialisation of the weight matrix is `W = init(out, in)`, calling the function
given to keyword `init`, with default [`glorot_uniform`](@ref Flux.glorot_uniform).
The weight matrix and/or the bias vector (of length `out`) may also be provided explicitly.

This layer differs from [`Dense`](@ref Flux.Dense) in that there is an extra nonlinear transformation
of the weight matrix before multiplication with the input.
"""
struct TransformedDense{F,FF,M<:AbstractMatrix,B}
    weight::M
    bias::B
    σ::F
    σ2::FF
    function Dense(
        W::M, bias=true, σ::F=identity, σ2::FF=identity
    ) where {M<:AbstractMatrix,F,FF}
        b = create_bias(W, bias, size(W, 1))
        return new{F,FF,M,typeof(b)}{W,b,σ,σ2}
    end
end

function TransformedDense(
    (in, out)::Pair{<:Integer,<:Integer},
    σ=identity,
    σ2=identity;
    init=Flux.glorot_uniform,
    bias=true,
)
    return TransformedDense(init(out, in), bias, σ, σ2)
end

@functor TransformedDense

function (a::TransformedDense)(x::AbstractVecOrMat)
    σ = NNlib.fast_act(a.σ, x)
    σ2 = NNlib.fast_act(a.σ2, x)
    return σ.(σ2.(a.weight) * x .+ a.bias)
end

function (a::TransformedDense)(x::AbstractArray)
    return reshape(a(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)
end

function Base.show(io::IO, l::TransformedDense)
    print(io, "TransformedDense(", size(l.weight, 2), " => ", size(l.weight, 1))
    l.σ == identity || print(io, ", ", l.σ)
    l.σ2 == identity || print(io, ", ", l.σ2)
    l.bias == false && print(io, "; bias=false")
    return print(io, ")")
end

