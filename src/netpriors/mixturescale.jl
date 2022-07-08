using Random
using StatsBase
using LinearAlgebra

"""
Scale mixture of Gaussians

# Fields
- `num_params_hyper::Int=0`: Number of hyper priors
- `nc::NetConstructor`: NetConstructor object
- `σ1`: Standard deviation of first Gaussian 
- `σ2`: Standard deviation of second Gaussian 
- `π1`: Weight/Probability of first Gaussian 
- `π2`: Weight/Probability of second Gaussian (`1-π1`)
"""
struct MixtureScalePrior{T, F} <: NetworkPrior
    num_params_hyper::Int
    nc::NetConstructor{T, F}
    σ1::T
    σ2::T
    π1::T
    π2::T
end
function MixtureScalePrior(nc::NetConstructor{T, F}, σ1::T, σ2::T, μ1::T) where {T, F}
    return MixtureScalePrior(0, nc, σ1, σ2, μ1, T(1) - μ1)
end

function (msp::MixtureScalePrior{T, F})(θnet::AbstractVector{T}, θhyper::AbstractVector{T}) where {T, F}
    n = length(θnet)
    return msp.π1 * logpdf(MvNormal(zeros(T, n), msp.σ1^2*I), θnet) + msp.π2 * logpdf(MvNormal(zeros(T, n), msp.σ2^2*I), θnet)
end

function sample_prior(msp::MixtureScalePrior{T, F}, rng::AbstractRNG = Random.GLOBAL_RNG) where {T, F}
    n = msp.nc.num_params_network
    θnet1 = rand(rng, MvNormal(zeros(T, n), msp.σ1^2 * I))
    θnet2 = rand(rng, MvNormal(zeros(T, n), msp.σ2^2 * I))
    take_1 = StatsBase.sample(0:1, StatsBase.ProbabilityWeights([msp.π2, msp.π1]), n)
    return take_1 .* θnet1 .+ (T(1) .- take_1) .* θnet2
end