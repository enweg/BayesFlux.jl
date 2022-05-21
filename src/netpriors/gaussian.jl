# Very simple prior. 
# θ_i ∼ Normal(0, σ0) for all network parameters θ_i
using Distributions
using Random

struct GaussianPrior{T, F} <: NetworkPrior
    num_params_hyper::Int
    nc::NetConstructor{T, F}
    σ0::T
end
function GaussianPrior(nc::NetConstructor{T, F}, σ0::T = T(1.0)) where {T, F}
    return GaussianPrior(0, nc, σ0)
end

function (gp::GaussianPrior{T, F})(θnet::Vector{T}, θhyper::Vector{T}) where {T, F}
    n = length(θnet)
    return logpdf(MvNormal(zeros(T, n), gp.σ0^2*I), θnet)
end

function sample_prior(gp::GaussianPrior{T, F}, rng::AbstractRNG = Random.GLOBAL_RNG) where {T, F}
    θnet = rand(rng, MvNormal(zeros(T, gp.nc.num_params_network), gp.σ0^2*I))
    return θnet
end
