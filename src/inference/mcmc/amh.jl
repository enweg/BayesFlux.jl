"""
Adaptive Metropolis Hastings as introduced in 

Haario, H., Saksman, E., & Tamminen, J. (2001). An adaptive Metropolis
algorithm. Bernoulli, 223-242.

# Fields

- `samples::Matix`: Matrix holding the samples. If sampling was stopped early,
  not all columns will represent samples. To figure out how many columns
  represent samples, check out `nsampled`.
- `nsampled::Int`: Number of samples obtained.
- `C0::Matrix`: Initial covariance matrix. 
- `Ct::Matrix`: Covariance matrix in iteration t 
- `t::Int`: Current time period
- `t0::Int`: When to start adaptig the covariance matrix? Covariance is adapted
  in a rolling window form. 
- `sd::T`: See the paper. 
- `ϵ::T`: Will be added to diagonal to prevent numerical non-pod-def problems.
  If you run into numerical problems, try increasing this values.
- `accepted::Vector{Bool}`: For each sample, indicating whether the sample was
  accepted (true) or the previous samples was chosen (false)

# Notes 

- Adaptive MH might not be suited if it is very costly to calculate the
  likelihood as this needs to be done for each sample on the full dataset. Plans
  exist to make this faster. 
- Works best when started at a MAP estimate. 
"""
mutable struct AdaptiveMH{T} <: MCMCState
    samples::Matrix{T}
    nsampled::Int
    C0::Matrix{T}
    Ct::Matrix{T}
    t::Int
    t0::Int
    sd::T
    ϵ::T
    accepted::Vector{Bool}
end

"""
    function AdaptiveMH(C0::Matrix{T}, t0::Int, sd::T, ϵ::T) where {T}

Construct an Adaptive MH sampler. 

# Arguments 

- `C0`: Initial covariance matrix. Can usually be chosen to be the identity
  matrix created using `diagm(ones(T, bnn.num_total_params))`
- `t0`: Lookback window for adaptation of covariance matrix. Also means that
  adaptation does not happen until at least `t0` samples were drawn.
- `sd`: See paper. 
- `ϵ`: Used to overcome numerical problems. 
"""
function AdaptiveMH(C0::Matrix{T}, t0::Int, sd::T, ϵ::T) where {T}
    return AdaptiveMH(Matrix{T}(undef, 1, 1), 0, C0, similar(C0), 1, t0, sd, ϵ, Bool[])
end

function initialise!(
    s::AdaptiveMH{T}, 
    θ::AbstractVector{T}, 
    nsamples::Int; 
    continue_sampling=false
) where {T}

    samples = Matrix{T}(undef, length(θ), nsamples)
    if continue_sampling
        samples[:, 1:s.nsampled] = s.samples[:, 1:s.nsampled]
    end
    t = continue_sampling ? s.nsampled + 1 : 1
    nsampled = continue_sampling ? s.nsampled : 0
    Ct = continue_sampling ? s.Ct : s.C0

    s.samples = samples
    s.nsampled = nsampled
    s.Ct = Ct
    s.t = t
    s.accepted = fill(false, nsamples)
end

function calculate_epochs(
    s::AdaptiveMH{T}, 
    nbatches, 
    nsamples; 
    continue_sampling=false
) where {T}

    n_newsamples = continue_sampling ? nsamples - s.nsampled : nsamples
    epochs = ceil(Int, n_newsamples / nbatches)
    return epochs
end

function update!(s::AdaptiveMH{T}, θ::AbstractVector{T}, bnn::BNN, ∇θ) where {T}
    if s.t > s.t0
        # Adapt covariance matrix. The paper actually states a more efficient way. 
        # TODO: implement the more efficient way.
        s.Ct = s.sd * cov(s.samples[:, s.nsampled-s.t0+1:s.nsampled]') + s.sd * s.ϵ * I
    end
    θprop = rand(MvNormal(θ, s.Ct))
    lMH = loglikeprior(bnn, θprop, bnn.x, bnn.y) - loglikeprior(bnn, θ, bnn.x, bnn.y)

    r = rand()
    if r < exp(lMH)
        s.samples[:, s.nsampled+1] = copy(θprop)
        s.accepted[s.nsampled+1] = true
        θ = θprop
    else
        s.samples[:, s.nsampled+1] = copy(θ)
    end

    s.nsampled += 1
    s.t += 1

    return θ
end
