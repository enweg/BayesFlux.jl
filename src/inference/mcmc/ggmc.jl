using LinearAlgebra

"""
Gradient Guided Monte Carlo

Proposed in Garriga-Alonso, A., & Fortuin, V. (2021). Exact langevin dynamics
with stochastic gradients. arXiv preprint arXiv:2102.01691.

# Fields 

- `samples::Matrix`: Matrix containing the samples. If sampling stopped early,
  then not all columns will actually correspond to samples. See `nsampled` to
  check how many samples were actually taken
- `nsampled::Int`: Number of samples taken. 
- `t::Int`: Total number of steps taken.
- `accepted::Vector{Bool}`: If true, sample was accepted; If false, proposed
  sample was rejected and previous sample was taken. 
- `β::T`: See paper. 
- `l::T`: Step-length; See paper.
- `sadapter::StepsizeAdapter`: A StepsizeAdapter. Default is
  `DualAveragingStepSize`
- `M::AbstractMatrix`: Mass Matrix
- `Mhalf::AbstractMatrix`: Lower triangual cholesky decomposition of `M`
- `Minv::AbstractMatrix`: Inverse mass matrix.
- `madapter::MassAdapter`: A MassAdapter
- `momentum::AbstractVector`: Last momentum vector
- `lMH::T`: log of Metropolis-Hastings ratio. 
- `steps::Int`: Number of steps to take before calculating MH ratio. 
- `current_step::Int`: Current step in the recurrent sequence 1, ..., `steps`. 
- `maxnorm::T`: Maximimum gradient norm. Gradients are being clipped if norm
  exceeds this value

"""
mutable struct GGMC{T} <: MCMCState
    samples::Matrix{T}
    nsampled::Int
    t::Int
    accepted::Vector{Int}

    β::T
    l::T
    sadapter::StepsizeAdapter

    M::AbstractMatrix{T}  # Mass Matrix
    Mhalf::AbstractMatrix{T}
    Minv::AbstractMatrix{T}
    madapter::MassAdapter

    momentum::AbstractVector{T}
    lMH::T

    steps::Int  # Delayed acceptance
    current_step::Int

    maxnorm::T  # maximum gradient norm
end


function GGMC(
    type=Float32; 
    β::T=0.55f0, 
    l::T=0.0001f0,
    sadapter::StepsizeAdapter=DualAveragingStepSize(l),
    madapter::MassAdapter=DiagCovMassAdapter(1000, 100), 
    steps::Int=1,
    maxnorm::T=5.0f0
) where {T}

    samples = Matrix{type}(undef, 1, 1)
    nsampled = 0
    t = 1
    accepted = Int[]
    M, Mhalf, Minv = diagm(ones(T, 1)), diagm(ones(T, 1)), diagm(ones(T, 1))
    momentum = zeros(T, 1)

    return GGMC(
        samples, 
        nsampled, 
        t, 
        accepted,
        β, 
        l, 
        sadapter, 
        M, 
        Mhalf, 
        Minv, 
        madapter,
        momentum, 
        T(0), 
        steps, 
        1, 
        maxnorm
    )
end

function initialise!(
    s::GGMC{T}, 
    θ::AbstractVector{T}, 
    nsamples; 
    continue_sampling=false
) where {T,S,M}

    samples = Matrix{T}(undef, length(θ), nsamples)
    accepted = zeros(Int, nsamples)
    if continue_sampling
        samples[:, 1:s.nsampled] = s.samples[:, 1:s.nsampled]
        accepted[1:s.nsampled] = s.accepted
    end
    t = continue_sampling ? s.t : 1
    nsampled = continue_sampling ? s.nsampled : 0

    s.samples = samples
    s.t = t
    s.nsampled = nsampled
    s.accepted = accepted
    s.current_step = 1

    n = length(θ)
    if !continue_sampling
        s.M, s.Mhalf, s.Minv = diagm(ones(T, n)), diagm(ones(T, n)), diagm(ones(T, n))
    end
    s.momentum = zeros(T, n)
end

function calculate_epochs(
    s::GGMC{T}, 
    nbatches, 
    nsamples; 
    continue_sampling=false
) where {T,S,M}

    n_newsamples = continue_sampling ? nsamples - s.nsampled : nsamples
    epochs = ceil(Int, n_newsamples / nbatches)
    return epochs
end


K(m, Minv) = 1 / 2 * m' * Minv * m

function update!(s::GGMC{T}, θ::AbstractVector{T}, bnn::BNN, ∇θ) where {T,S,M}

    γ = -sqrt(length(bnn.y) / s.l) * log(s.β)
    h = sqrt(s.l / length(bnn.y))
    a = exp(-γ * h)

    if s.current_step == 1
        s.lMH = -loglikeprior(bnn, θ, bnn.x, bnn.y)
    end

    v, g = ∇θ(θ)
    g = -g
    g = clip_gradient!(g; maxnorm=s.maxnorm)

    h = sqrt(h)
    momentum = s.momentum
    momentum14 = sqrt(a) * momentum + sqrt((T(1) - a)) * s.Mhalf * randn(length(θ))
    momentum12 = momentum14 .- h / T(2) * g
    θ .+= h * s.Minv * momentum12

    if any(isnan.(θ))
        error("NaN in θ")
    end

    v, g = ∇θ(θ)
    g = -g
    g = clip_gradient!(g; maxnorm=s.maxnorm)

    momentum34 = momentum12 - h / T(2) * g
    s.momentum = sqrt(a) * momentum34 + sqrt((1 - a)) * s.Mhalf * rand(Normal(T(0), T(1)), length(θ))

    s.lMH += K(momentum34, s.Minv) - K(momentum14, s.Minv)
    s.samples[:, s.t] = copy(θ)
    s.nsampled += 1

    if s.current_step == s.steps && s.t > s.steps
        s.lMH += loglikeprior(bnn, θ, bnn.x, bnn.y)

        s.l = s.sadapter(s, min(exp(s.lMH), 1))
        s.Minv = s.madapter(s, θ, bnn, ∇θ)
        s.M = inv(s.Minv)
        s.Mhalf = cholesky(s.M; check=false).L

        r = rand()
        if r < min(exp(s.lMH), 1) #|| s.nsampled == 1 
            # accepting 
            s.accepted[(s.nsampled-s.steps+1):s.nsampled] = ones(Int, s.steps)
        else
            # rejecting
            s.samples[:, (s.nsampled-s.steps+1):s.nsampled] = hcat(fill(copy(s.samples[:, s.nsampled-s.steps]), s.steps)...)
            θ = copy(s.samples[:, s.nsampled-s.steps])
        end
    end

    s.t += 1
    s.current_step = (s.current_step == s.steps) ? 1 : s.current_step + 1

    return θ
end



