
"""
Stochastic Gradient Nose-Hoover Thermostat as proposed in 

Proposed in Leimkuhler, B., & Shang, X. (2016). Adaptive thermostats for noisy
gradient systems. SIAM Journal on Scientific Computing, 38(2), A712-A736.

This is similar to SGNHT as proposed in 
Ding, N., Fang, Y., Babbush, R., Chen, C., Skeel, R. D., & Neven, H. (2014).
Bayesian sampling using stochastic gradient thermostats. Advances in neural
information processing systems, 27.

# Fields
- `samples::Matrix`: Containing the samples 
- `nsampled::Int`: Number of samples taken so far. Can be smaller than
  `size(samples, 2)` if sampling was interrupted. 
- `p::AbstractVector`: Momentum
- `xi::Number`: Thermostat
- `l::Number`: Stepsize; This often is in the 0.001-0.1 range.
- `σA::Number`: Diffusion factor; If the stepsize is small, this should be larger than 1.
- `μ::Number`: Free parameter in thermostat. Defaults to 1.
- `t::Int`: Current step count
- `kinetic::Vector`: Keeps track of the kinetic energy. Goal of SGNHT is to have
  the average close to one

# Notes
- Does not clip gradients since that seems to cause to a failure of the method.
- TODO: why does this happen
"""
mutable struct SGNHTS{T} <: MCMCState
    samples::Matrix{T}
    nsampled::Int
    p::AbstractVector{T}
    xi::T
    l::T 
    σA::T 
    μ::T
    t::Int

    kinetic::Vector{T}
end
SGNHTS(l::T, σA::T = T(1); xi = T(1), μ = T(1)) where {T} = SGNHTS(Matrix{T}(undef, 0, 0), 0, T[], xi, l, σA, μ, 1, T[])

function initialise!(s::SGNHTS{T}, θ::AbstractVector{T}, nsamples::Int; continue_sampling = false) where {T}
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

function calculate_epochs(s::SGNHTS{T}, nbatches, nsamples; continue_sampling = false) where {T}
    n_newsamples = continue_sampling ? nsamples - s.nsampled : nsamples
    epochs = ceil(Int, n_newsamples/nbatches)
    return epochs
end

function update!(s::SGNHTS{T}, θ::AbstractVector{T}, bnn::BNN, ∇θ) where {T}

    v, g = ∇θ(θ)
    # ng = norm(g)
    # maxnorm = T(5)
    # g = ng > maxnorm ? maxnorm.*g./ng : g
    # They work with potential energy which is negative loglike so negate
    # gradient 
    g = -g
    if any(isnan.(g))
        error("NaN in g")
    end

    n = length(θ)
    s.p = s.p - s.l/T(2)*g
    θ = θ + s.l/T(2)*s.p 
    s.xi = s.xi + s.l/(T(2)*s.μ) * (s.p'*s.p - n)  # We need kT = 1
    if s.xi != 0
        s.p = exp(-s.xi*s.l)*s.p + s.σA*sqrt((1 - exp(-T(2)*s.xi*s.l))/(T(2)*s.xi)) * randn(T, n)
    else
        s.p = s.p + sqrt(s.l)*s.σA*randn(T, n)
    end
    s.xi = s.xi + s.l/(T(2)*s.μ)*(s.p'*s.p - n)
    θ = θ + s.l/T(2)*s.p

    if any(isnan.(θ))
        error("NaN in θ")
    end

    s.samples[:, s.t] = copy(θ)
    s.kinetic[s.t] = 1/n * s.p's.p

    s.t += 1
    s.nsampled += 1
    return θ
end