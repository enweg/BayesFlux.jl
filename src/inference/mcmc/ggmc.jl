# Gradient Guided Monte Carlo
using LinearAlgebra
"""

Used to adjust the stepsize of GGMC. 

- Must be callable and take `s::MCMCState` and the current Metropolis-Hastings
  ratio and must return a new stepsize. We are working with `l` here instead of
  with `h`.

"""
abstract type StepsizeAdapter end

"""
Does not change the stepsize. 
"""
struct StepsizeConstantAdapter <: StepsizeAdapter
    adaptation_steps::Int
end
(ladapter::StepsizeConstantAdapter)(s::MCMCState, mh::T) where {T} = s.l

"""
Use simple stochastic optimisation to adapt steplength.
"""
mutable struct StepsizeStochasticOptAdapter{T} <: StepsizeAdapter
    κ::T 
    goal_accept_rate::T
    t::Int
    adaptation_steps::Int
end
StepsizeStochasticOptAdapter(κ = 0.55f0, goal_accept_rate = 0.65f0, adaptation_steps = 1000) = StepsizeStochasticOptAdapter(κ, goal_accept_rate, 1, adaptation_steps)
function (ladapter::StepsizeStochasticOptAdapter)(s::MCMCState, lmh::T) where {T}
    ladapter.t > ladapter.adaptation_steps && return s.l
    @unpack κ, goal_accept_rate, t = ladapter 
    H = goal_accept_rate - min(exp(lmh), T(1)) 
    η = t^(-κ)
    s.l = exp(log(s.l) - η*H)
    ladapter.t == ladapter.adaptation_steps && @info "Final l = $(s.l)"
    ladapter.t += 1
    return s.l
end

"""

Adaptation of Mass matrix.

- Must be callable and take `s::MCMCState`, θ and ∇θ, thet current parameter vector
  and its gradient. Must return M, Mhalf, Minv in that order.

"""
abstract type MassAdapter end

"""
Always returns the identity matrix.
"""
struct MassIdentityAdapter <: MassAdapter
    adaptation_steps::Int
end
function (madapter::MassIdentityAdapter)(s::MCMCState, θ::AbstractVector{T}, g::AbstractVector{T}) where {T} 
    n = length(g) 
    return Diagonal(diagm(ones(T, n))), Diagonal(diagm(ones(T, n))), Diagonal(diagm(ones(T, n)))
end

"""
RMSProp Mass matrix adaptation
"""
mutable struct MassRMSPropAdapter{T} <: MassAdapter
    ν::AbstractVector{T}
    β::T
    λ::T
    adapt_steps::Int
    t::Int
end
MassRMSPropAdapter(size, adapt_steps = 1000; β::T = 0.99f0, λ::T = 1f-8) where {T} = MassRMSPropAdapter(zeros(T, size), β, λ, adapt_steps, 1)
function (madapter::MassRMSPropAdapter)(s::MCMCState, θ::AbstractVector{T}, g::AbstractVector{T}) where {T}
    @unpack ν, β, λ, adapt_steps, t = madapter
    t > adapt_steps && return s.M, s.Mhalf, s.Minv

    madapter.ν .= β*ν .+ (1-β)*g.*g
    G = Diagonal(diagm(1 ./ (λ .+ sqrt.(ν))))
    Minv = G
    M = Diagonal(diagm(1.0 ./ diag(G)))
    Mhalf = Diagonal(sqrt.(M))
    s.M, s.Mhalf, s.Minv = M, Mhalf, Minv

    t == adapt_steps && @info "Finished adapting Mass Matrix M = $M"
    madapter.t += 1
    return M, Mhalf, Minv
end

"""
Diagonal Mass adapter via variance
TODO: Very inefficient currently
"""
mutable struct MassVarianceAdapter <: MassAdapter
    start_after::Int
    adapt_steps::Int
    t::Int
end
MassVarianceAdapter(start_after = 100, adapt_steps = 500) = MassVarianceAdapter(start_after, adapt_steps, 1)
function (madapter::MassVarianceAdapter)(s::MCMCState, θ::AbstractVector{T}, g::AbstractVector{T}) where {T}
    @unpack start_after, adapt_steps, t = madapter
    if t <= start_after || t > start_after + adapt_steps
        madapter.t += 1
        return s.M, s.Mhalf, s.Minv
    end
    prop_Minv = Diagonal(diagm(vec(var(s.samples; dims = 2))))
    prop_M = Diagonal(diagm(T(1) ./ diag(prop_Minv)))
    prop_Mhalf = sqrt(prop_M)
    if !any(isnan.(prop_M) .|| isinf.(prop_M) .|| isinf.(prop_Minv))
        print("adapting M")
        s.M, s.Mhalf, s.Minv = prop_M, prop_Mhalf, prop_Minv
    end
    t == start_after + adapt_steps && @info "Finished adapting Mass Matrix M = $(s.M)"
    madapter.t += 1
    return s.M, s.Mhalf, s.Minv
end


"""
Gradient Guided Monte Carlo
"""
mutable struct GGMC{T, S<:StepsizeAdapter, M<:MassAdapter} <: MCMCState
    θ::AbstractVector{T}
    samples::Matrix{T}
    nsampled::Int
    t::Int 
    accepted::Vector{Int}

    β::T
    l::T
    temp::T
    stepsize_adapter::S
    M::AbstractMatrix{T}  # Mass Matrix
    Mhalf::AbstractMatrix{T}
    Minv::AbstractMatrix{T}
    M_adapter::M
    momentum::AbstractVector{T}
    lMH::T

    steps::Int  # Delayed acceptance
    current_step::Int
end


function GGMC(type = Float32, ;β::T = 0.55f0, l::T = 0.0001f0, temp::T = 1.0f0, 
    stepsize_adapter::StepsizeAdapter = StepsizeConstantAdapter(1000), 
    M_adapter::MassAdapter = MassIdentityAdapter(1000), steps::Int = 1) where {T}

    θ = type[]
    samples = Matrix{type}(undef, 1, 1)
    nsampled = 0
    t = 1
    accepted = Int[]
    M, Mhalf, Minv = diagm(ones(T, 1)), diagm(ones(T, 1)), diagm(ones(T, 1)) 
    momentum = zeros(T, 1)

    return GGMC(θ, samples, nsampled, t, accepted, 
        β, l, temp, stepsize_adapter, M, Mhalf, Minv, M_adapter, 
        momentum, T(0), steps, 1)
end

function initialise!(s::GGMC{T, S, M}, θ::AbstractVector{T}, nsamples; continue_sampling = false) where {T, S, M}
    samples = Matrix{T}(undef, length(θ), nsamples) 
    if continue_sampling
        samples[:, 1:s.nsampled] = s.samples[:, 1:s.nsampled]
    end
    t = continue_sampling ? s.t : 1
    nsampled = continue_sampling ? s.nsampled : 0

    s.θ = θ
    s.samples = samples
    s.t = t
    s.nsampled = nsampled
    accepted = zeros(Int, nsamples)
    if continue_sampling
        accepted[1:s.nsampled] = s.accepted
    end
    s.accepted = accepted 

    n = length(θ)
    s.M, s.Mhalf, s.Minv = diagm(ones(T, n)), diagm(ones(T, n)), diagm(ones(T, n))
    s.momentum = zeros(T, n)
end

function calculate_epochs(s::GGMC{T, S, M}, nbatches, nsamples; continue_sampling = false) where {T, S, M}
    n_newsamples = continue_sampling ? nsamples - s.nsampled : nsamples
    epochs = ceil(Int, n_newsamples / nbatches)
    return epochs
end


K(m, Minv) = 1/2 * m' * Minv * m

function update!(s::GGMC{T, S, M}, θ::AbstractVector{T}, bnn::BNN, ∇θ) where {T, S, M}

    γ = -sqrt(length(bnn.y)/s.l)*log(s.β)
    h = sqrt(s.l / length(bnn.y))
    a = exp(-γ * h)

    if s.current_step == 1
        s.lMH = -loglikeprior(bnn, θ, bnn.x, bnn.y)
    end

    v, g = ∇θ(θ)
    g = -g
    g = clip_gradient_value!(g, T(15.0))

    h = sqrt(h)
    momentum = s.momentum
    momentum14 = sqrt(a)*momentum + sqrt((T(1)-a)*s.temp)*s.Mhalf*randn(length(θ))
    # momentum12 = momentum14 .- sqrt(h)/T(2) * g
    # θ .+= sqrt(h)*s.Minv*momentum12 
    momentum12 = momentum14 .- h/T(2) * g
    θ .+= h*s.Minv*momentum12 

    v, g = ∇θ(θ)
    g = -g
    g = clip_gradient_value!(g, T(15.0))
    momentum34 = momentum12 - h/T(2)*g
    s.momentum = sqrt(a)*momentum34 + sqrt((1-a)*s.temp)*s.Mhalf*rand(Normal(T(0), T(1)), length(θ))

    s.lMH += K(momentum34, s.Minv) - K(momentum14, s.Minv)
    s.samples[:, s.t] = copy(θ)
    s.nsampled += 1

    if s.current_step == s.steps && s.t > s.steps
        s.lMH += loglikeprior(bnn, θ, bnn.x, bnn.y)

        s.l = s.stepsize_adapter(s, s.lMH)
        s.M, s.Mhalf, s.Minv = s.M_adapter(s, θ, g) 

        r = rand()
        if r < min(exp(s.lMH), 1) #|| s.nsampled == 1 
            # accepting 
            s.accepted[(s.nsampled - s.steps + 1):s.nsampled] = ones(Int, s.steps) 
        else
            # rejecting
            s.samples[:, (s.nsampled - s.steps + 1):s.nsampled] = hcat(fill(copy(s.samples[:, s.nsampled - s.steps]), s.steps)...)
            θ = copy(s.samples[:, s.nsampled - s.steps])
        end
    end

    s.t += 1
    s.current_step = (s.current_step == s.steps) ? 1 : s.current_step + 1
    s.θ = θ

    return θ
end



