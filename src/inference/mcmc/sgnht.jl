"""
Stochastic Gradient Nose-Hoover Thermostat as proposed in 

Ding, N., Fang, Y., Babbush, R., Chen, C., Skeel, R. D., & Neven, H. (2014).
Bayesian sampling using stochastic gradient thermostats. Advances in neural
information processing systems, 27.

# Fields
- `samples::Matrix`: Containing the samples 
- `nsampled::Int`: Number of samples taken so far. Can be smaller than
  `size(samples, 2)` if sampling was interrupted. 
- `p::AbstractVector`: Momentum
- `xi::Number`: Thermostat
- `l::Number`: Stepsize
- `A::Number`: Diffusion factor
- `t::Int`: Current step count
- `kinetic::Vector`: Keeps track of the kinetic energy. Goal of SGNHT is to have
  the average close to one
"""
mutable struct SGNHT{T} <: MCMCState
    samples::Matrix{T}
    nsampled::Int
    p::AbstractVector{T}
    xi::T
    l::T
    A::T
    t::Int

    kinetic::Vector{T}
end

SGNHT(l::T, A::T=T(1); xi=T(0)) where {T} = SGNHT(Matrix{T}(undef, 0, 0), 0, T[], xi, l, A, 1, T[])

function initialise!(
    s::SGNHT{T}, 
    θ::AbstractVector{T}, 
    nsamples::Int; 
    continue_sampling=false
) where {T}

    samples = Matrix{T}(undef, length(θ), nsamples)
    kinetic = Vector{T}(undef, nsamples)
    if continue_sampling
        samples[:, 1:s.nsampled] = s.samples[:, 1:s.nsampled]
        kinetic[1:s.nsampled] = s.kinetic
    end
    t = continue_sampling ? s.nsampled + 1 : 1
    nsampled = continue_sampling ? s.nsampled : 0

    s.samples = samples
    s.kinetic = kinetic
    s.nsampled = nsampled
    s.t = t
    s.p = zero(θ)

end

function calculate_epochs(s::SGNHT{T}, nbatches, nsamples; continue_sampling=false) where {T}
    n_newsamples = continue_sampling ? nsamples - s.nsampled : nsamples
    epochs = ceil(Int, n_newsamples / nbatches)
    return epochs
end

function update!(s::SGNHT{T}, θ::AbstractVector{T}, bnn::BNN, ∇θ) where {T}

    v, g = ∇θ(θ)
    # They work with potential energy which is negative loglike so negate
    # gradient 
    g = -g
    if any(isnan.(g))
        error("NaN in g")
    end

    s.p = s.p - s.xi * s.l * s.p - s.l * g + sqrt(s.l * 2 * s.A) * rand(MvNormal(zero.(θ), one.(θ)))
    θ = θ + s.l * s.p
    n = length(θ)
    kinetic = T(1) / n * s.p' * s.p
    s.kinetic[s.t] = kinetic
    s.xi = s.xi + (kinetic - 1) * s.l

    if any(isnan.(θ))
        error("NaN in θ")
    end

    s.samples[:, s.t] = copy(θ)

    s.t += 1
    s.nsampled += 1
    return θ
end