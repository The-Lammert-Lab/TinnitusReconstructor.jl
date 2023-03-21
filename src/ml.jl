
using Flux
using Flux: create_bias
using Functors
import Base: show
using LinearAlgebra
using Random: AbstractRNG, default_rng
using Tullio

"""
    mmd(x, y; σ=1)

Compute the maximum mean discrepancy (MMD)
between `x` and `y` using a Gaussian kernel
with standard deviation parameter `σ`.

# Examples
```jldoctest
julia> mmd(1, 1)
0.0

julia> mmd(1, 2; σ=1)
0.7869386805747332

julia> mmd(1, 2; σ=2)
0.2350061948308091
```
"""
function mmd(x, y; σ=1)
    T = eltype(x)
    M = length(x)
    N = length(y)
    pre = T(-1/(2 * σ * σ))

    # Here the macro can see the function & compute symbolic derivatives:
    @tullio running_total := exp(pre * abs2(x[i] - x[j]))
    mmd = running_total / M^2

    @tullio running_total2 := exp(pre * abs2(x[i] - y[j]))
    mmd -= 2 * running_total2 / (M * N)

    @tullio running_total3 := exp(pre * abs2(y[i] - y[j]))
    mmd += running_total3 / N^2

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
    return exp(
        -one(typeof(x)) / (oftype(x / 1, 2) * oftype(x / 1, σ) * oftype(x / 1, σ)) *
        abs2(x - y)
    )
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
    function TransformedDense(
        W::M, bias=true, σ::F=identity, σ2::FF=identity
    ) where {M<:AbstractMatrix,F,FF}
        b = create_bias(W, bias, size(W, 1))
        return new{F,FF,M,typeof(b)}(W, b, σ, σ2)
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

@doc raw"""
    scaled_uniform([rng = Flux.default_rng_value()], size...; gain = 1) -> Array
    scaled_uniform([rng], kw...) -> Function

Return an `Array{Float32}` of the given `size` containing random numbers drawn from a uniform distribution
on the interval ``gain * [0, 1]``.

# Examples

julia> round.(extrema(scaled_uniform(100, 10; gain=2π)), digits=3)
(0.004, 6.282)

julia> scaled_uniform(5) |> summary
"5-element Vector{Float32}"
"""
function scaled_uniform(rng::AbstractRNG, dims::Integer...; gain::Real=1)
    return (rand(rng, Float32, dims...)) .* gain * 1.0f0
end
function scaled_uniform(dims::Integer...; kw...)
    return scaled_uniform(Flux.default_rng_value(), dims...; kw...)
end
function scaled_uniform(rng::AbstractRNG=Flux.default_rng_value(); init_kwargs...)
    return (dims...; kwargs...) -> scaled_uniform(rng, dims...; init_kwargs..., kwargs...)
end

# TODO: fix the regularization
@doc """
    mmd_loss(x, x̂; σs=[1])

Compute the mean maximum discrepancy loss
with a Gaussian kernel.
`σs` is a list of kernel sizes (standard deviations)
that the loss is summed over.

# Examples
```jldoctest
julia> mmd_loss(1, 1; σs=[1, 2, 3])
0.0

julia> mmd_loss(1, 1)
0.0

julia> mmd_loss(1, 2)
0.7869386805747332
```

# See Also

* [mmd](@ref mmd)
"""
function mmd_loss(x, x̂; σs=[1])
    return sum(mmd(x, x̂; σ=σ) for σ in σs)
end

"""
    generate_data(n_samples::T, m::T, n::T, p::T) where {T<:Integer}

Generate `n_samples` training samples of length `n`
with sparsity `p`.
The size of `H` is `n × n_samples`
and the size of `U` is `m x n_samples`.
Note that these samples are sparse in the standard basis (identity matrix).
"""
function generate_data(rng::AbstractRNG, n_samples::T, m::T, n::T, p::T) where {T<:Integer}
    # Generate a Gaussian random matrix
    H = randn(rng, Float32, n, n_samples) ./ p
    # Set all but p indices in each row to zero
    for h in eachcol(H)
        indices = sample(1:n, n - p; replace=false)
        h[indices] .= 0
    end
    # Rescale
    H /= sqrt(norm(H) / n_samples)

    # Compute the label data
    U = randn(rng, Float32, m, n_samples)
    for u in eachcol(U)
        u .= u / norm(u)
    end
    return Float32.(H), Float32.(U)
end
function generate_data(n_samples::T, m::T, n::T, p::T) where {T <: Integer}
    generate_data(default_rng(), n_samples, m, n, p)
end