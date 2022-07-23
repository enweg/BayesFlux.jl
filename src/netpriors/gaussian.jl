# Very simple prior. 
# θ_i ∼ Normal(0, σ0) for all network parameters θ_i
using Distributions
using Random

struct GaussianPrior{T, F} <: NetworkPrior
    num_params_hyper::Int
    nc::NetConstructor{T, F}
    σ0::T
end
"""

Use a Gaussian prior for all network parameters. This means that we do not allow
for any correlations in the network parameters in the prior. 

# Arguments

- `nc`: obtained using [`destruct`](@ref)
- `σ0`: standard deviation of prior
"""
function GaussianPrior(nc::NetConstructor{T, F}, σ0::T = T(1.0)) where {T, F}
    return GaussianPrior(0, nc, σ0)
end

function (gp::GaussianPrior{T, F})(θnet::AbstractVector{T}, θhyper::AbstractVector{T}) where {T, F}
    n = length(θnet)
    return logpdf(MvNormal(zeros(T, n), gp.σ0^2*I), θnet)
end

function sample_prior(gp::GaussianPrior{T, F}, rng::AbstractRNG = Random.GLOBAL_RNG) where {T, F}
    θnet = rand(rng, MvNormal(zeros(T, gp.nc.num_params_network), gp.σ0^2*I))
    return θnet
end
