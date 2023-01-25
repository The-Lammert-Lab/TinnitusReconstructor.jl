"""
This script is a test to implement
the maximum mean discrepancy optimization
problem for designing a compressive sensing matrix
with structural constraints
by Koller & Utschick 2022.
"""

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
    gaussian_kernel(x, y, σ=1)

Compute the gaussian kernel for `x` and `y`.
This is the function
``k_\sigma : \mathbb{R}^{2m} \times \mathbb{R}^{2m} \rightarrow \mathbb{R}, (x, y) \mapsto k_\sigma (x, y) = \exp \left ( - \frac{1}{2\sigma^2} ||x-y||^2 \right )`` 

# Examples
```jldoctest
julia> gaussian_kernel(1, 1)
1.0
```
"""
function gaussian_kernel(x, y, σ=1)
    return @. exp(-1 / (2 * σ^2) * abs(x - y)^2)
end

"""
    phase_to_mm(Φ)

Convert a matrix of phases `Φ` to a measurement matrix via
``\frac{1}{\sqrt{m}} \exp(i \Phi)``.
"""
phase_to_mm(Φ) = 1 / sqrt(size(Φ, 1)) * exp(1im * Φ)

@doc raw"""
    stk(z)

Stack real and imaginary parts of a complex vector `z`
in a real vector `stk(z)`:
``\mathrm{stk} : \mathbb{C}^m \rightarrow \mathbb{R}^{2m}, z \mapsto \mathrm{stk}(z) = \left[\mathcal{R}(z)^{\mathrm{T}}, \mathcal{I}(z)^{\mathrm{T}} \right]^{\mathrm{T}}
"""
function stk(z)
    return vcat(vec(real(z)'), vec(imag(z)'))
end