# implementation of stochastic gradient langevin dynamics

"""
    stepsize(a, b, γ, t)

Calculate the stepsize. 

Calculates the next stepsize according to ϵₜ = a(b+t)^(-γ). This is a commonly used 
stepsize schedule that meets the criteria ∑ϵₜ = ∞, ∑ϵₜ² < ∞. 

"""
stepsize(a, b, γ, t) = a * (b + t)^(-γ)

"""
Stochastic Gradient Langevin Dynamics as proposed in Welling, M., & Teh, Y. W.
(n.d.). Bayesian Learning via Stochastic Gradient Langevin Dynamics. 8.

# Fields

-  `θ::AbstractVector`: Current sample
- `samples::Matrix`: Matrix of samples. Not all columns will be actual samples if sampling was stopped early. See `nsampled` for the actual number of samples taken. 
- `nsampled::Int`: Number of samples taken
- `min_stepsize::T`: Stop decreasing the stepsize when it is below this value. 
- `didinform::Bool`: Flag keeping track of whether we informed user that `min_stepsize` was reached. 
- `stepsize_a::T`: See `stepsize`
- `stepsize_b::T`: See `stepsize`
- `stepsize_γ::T`: See `stepsize`
"""
mutable struct SGLD{T} <: MCMCState
    θ::AbstractVector{T}
    samples::Matrix{T}
    nsampled::Int
    t::Int
    min_stepsize::T
    didinform::Bool

    stepsize_a::T
    stepsize_b::T
    stepsize_γ::T

end
function SGLD(type=Float32; stepsize_a=0.1f0, stepsize_b=1.0f0, stepsize_γ=0.55f0, min_stepsize=Float32(-Inf))
    return SGLD(type[], Matrix{type}(undef, 1, 1), 0, 1, min_stepsize, false, stepsize_a, stepsize_b, stepsize_γ)
end

function initialise!(s::SGLD{T}, θ::AbstractVector{T}, nsamples::Int; continue_sampling=false) where {T}
    samples = Matrix{T}(undef, length(θ), nsamples)
    if continue_sampling
        samples[:, 1:s.nsampled] = s.samples[:, 1:s.nsampled]
    end
    t = continue_sampling ? s.nsampled + 1 : 1
    nsampled = continue_sampling ? s.nsampled : 0

    s.θ = θ
    s.samples = samples
    s.t = t
    s.nsampled = nsampled
end

function calculate_epochs(s::SGLD{T}, nbatches, nsamples; continue_sampling=false) where {T}
    n_newsamples = continue_sampling ? nsamples - s.nsampled : nsamples
    epochs = ceil(Int, n_newsamples / nbatches)
    return epochs
end

function update!(s::SGLD{T}, θ::AbstractVector{T}, bnn::BNN, ∇θ) where {T}
    α = stepsize(s.stepsize_a, s.stepsize_b, s.stepsize_γ, s.t)
    α < s.min_stepsize && !s.didinform && @info "Using minstepsize=$(s.min_stepsize) from now."
    α = max(α, s.min_stepsize)

    v, g = ∇θ(θ)
    g = clip_gradient_value!(g, 15)
    g = α / T(2) .* g .+ sqrt(α) * randn(T, length(θ))
    θ .+= g

    s.samples[:, s.t] = copy(θ)
    s.nsampled += 1
    s.t += 1
    s.θ = θ

    return θ
end

