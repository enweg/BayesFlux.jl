using LinearAlgebra

"""
Standard Hamiltonian Monte Carlo (Hybrid Monte Carlo).

Allows for the use of stochastic gradients, but the validity of doing so is not clear. 

This is motivated by parts of the discussion in 
Neal, R. M. (1996). Bayesian Learning for Neural Networks (Vol. 118). Springer
New York. https://doi.org/10.1007/978-1-4612-0745-0

Code was partially adapted from
https://colindcarroll.com/2019/04/11/hamiltonian-monte-carlo-from-scratch/


# Fields
- `samples::Matrix`: Samples taken
- `nsampled::Int`: Number of samples taken. Might be smaller than
  `size(samples)` if sampling was interrupted.
- `θold::AbstractVector`: Old sample. Kept for rejection step.
- `momentum::AbstractVector`: Momentum variables
- `momentumold::AbstractVector`: Old momentum variables. Kept for rejection
  step.
- `t::Int`: Current step.
- `path_len::Int`: Number of leapfrog steps. 
- `current_step::Int`: Current leapfrog step.
- `accepted::Vector{Bool}`: Whether a draw in `samples` was a accepted draw or
  rejected (in which case it is the same as the previous one.)
- `sadapter::StepsizeAdapter`: Stepsize adapter giving the stepsize in each
  iteration.
- `l`: Stepsize.
- `madapter::MassAdapter`: Mass matrix adapter giving the inverse mass matrix in
  each iteration.
- `Minv::AbstractMatrix`: Inverse mass matrix
- `maxnorm::T`: Maximimum gradient norm. Gradients are being clipped if norm
  exceeds this value
"""
mutable struct HMC{T} <: MCMCState
    samples::Matrix{T}
    nsampled::Int

    θold::AbstractVector{T}
    momentum::AbstractVector{T}
    momentumold::AbstractVector{T}

    t::Int
    path_len::Int  # Leapfrog steps
    current_step::Int  # which step of leapfrog steps
    accepted::Vector{Bool}

    sadapter::StepsizeAdapter
    l::T
    madapter::MassAdapter
    Minv::AbstractMatrix{T}

    maxnorm::T  # maximum gradient norm
end

function HMC(
    l::T, 
    path_len::Int;
    sadapter=DualAveragingStepSize(l),
    madapter=FullCovMassAdapter(1000, 100), 
    maxnorm::T=5.0f0
) where {T}

    return HMC(Matrix{T}(undef, 1, 1), 0, T[], T[], T[], 1, path_len, 1, Bool[],
        sadapter, l, madapter, Matrix{T}(undef, 0, 0), maxnorm)
end

function initialise!(
    s::HMC{T}, 
    θ::AbstractVector{T}, 
    nsamples::Int; 
    continue_sampling=false
) where {T}

    samples = Matrix{T}(undef, length(θ), nsamples)
    accepted = fill(false, nsamples)
    if continue_sampling
        samples[:, 1:s.nsampled] = s.samples[:, 1:s.nsampled]
        accepted[1:s.nsampled] = s.accepted
    end
    t = continue_sampling ? s.nsampled + 1 : 1
    nsampled = continue_sampling ? s.nsampled : 0

    s.samples = samples
    s.nsampled = nsampled
    s.t = t
    s.accepted = accepted
    s.θold = copy(θ)
    s.momentum = zero(θ)
    s.momentumold = zero(θ)

    s.l = s.sadapter.l
    s.Minv = size(s.madapter.Minv, 1) == 0 ? Diagonal(one.(θ)) : s.madapter.Minv
end

function calculate_epochs(s::HMC{T}, nbatches, nsamples; continue_sampling=false) where {T}
    n_newsamples = continue_sampling ? nsamples - s.nsampled : nsamples
    epochs = ceil(Int, n_newsamples * s.path_len / nbatches)
    return epochs
end

function half_moment_update!(s::HMC{T}, θ::AbstractVector{T}, ∇θ) where {T}
    v, g = ∇θ(θ)
    g = -g  # everyone else works with negative loglikeprior 
    # Clipping
    # TODO: expose this to the user
    g = clip_gradient!(g; maxnorm=s.maxnorm)
    s.momentum .-= s.l * g / T(2)
end

function update!(s::HMC{T}, θ::AbstractVector{T}, bnn::BNN, ∇θ) where {T}

    moment_dist = MvNormal(zero(θ), s.Minv)

    if s.current_step == 1
        # New Leapfrog 
        s.momentum = rand(moment_dist)
        s.θold = copy(θ)
        s.momentumold = copy(s.momentum)
        half_moment_update!(s, θ, ∇θ)
    elseif s.current_step < s.path_len
        # Leapfrog loop 
        θ .+= s.l * s.momentum
        half_moment_update!(s, θ, ∇θ)
    else
        # Finishing off leapfrog and accept / reject
        θ .+= s.l * s.momentum
        half_moment_update!(s, θ, ∇θ)

        # moment flip
        s.momentum = -s.momentum

        start_mh = -loglikeprior(bnn, s.θold, bnn.x, bnn.y) - logpdf(moment_dist, s.momentumold)
        end_mh = -loglikeprior(bnn, θ, bnn.x, bnn.y) - logpdf(moment_dist, s.momentum)
        lMH = start_mh - end_mh
        lr = log(rand())

        s.l = s.sadapter(s, min(exp(lMH), 1))
        s.Minv = s.madapter(s, θ, bnn, ∇θ)

        if lr < lMH
            # accepting
            s.samples[:, s.nsampled+1] = copy(θ)
            s.accepted[s.nsampled+1] = true
        else
            # rejecting
            s.samples[:, s.nsampled+1] = copy(s.θold)
            θ = copy(s.θold)
        end
        s.nsampled += 1
    end

    s.current_step = s.current_step == s.path_len ? 1 : s.current_step + 1
    s.t += 1

    return θ
end
