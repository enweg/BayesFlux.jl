using Flux, Distributions, Random
using Zygote


struct BNN{Tx, Ty, L, P, I} 
    x::Tx
    y::Ty
    like::L 
    prior::P 
    init::I
    
    start_θnet::Int
    start_θhyper::Int
    start_θlike::Int
    num_total_params::Int
end

"""
    BNN(x, y, like::BNNLikelihood, prior::NetworkPrior, init::BNNInitialiser)

Create a Bayesian Neural Network. 

# Arguments

- `x`: Explanatory data
- `y`: Dependent variables
- `like`: A likelihood
- `prior`: A prior on network parameters
- `init`: An initilialiser

"""
function BNN(x, y, like::BNNLikelihood, prior::NetworkPrior, init::BNNInitialiser)
    start_θnet = 1
    start_θhyper = start_θnet + like.nc.num_params_network
    start_θlike = prior.num_params_hyper == 0 ? start_θhyper : start_θhyper + prior.num_params_hyper - 1
    num_total_params = like.nc.num_params_network + like.num_params_like + prior.num_params_hyper
    return BNN(x, y, like, prior, init, start_θnet, start_θhyper, start_θlike, num_total_params)
end

function split_params(bnn::B, θ::Vector{T}) where {B<:BNN, T}
    θnet = @view θ[1:(bnn.start_θhyper-1)]
    θhyper = bnn.prior.num_params_hyper == 0 ? eltype(θ)[] : @view θ[bnn.start_θhyper:(bnn.start_θlike-1)]
    θlike = @view θ[bnn.start_θlike:end]
    return θnet, θhyper, θlike
end

"""
Obtain the log of the unnormalised posterior.
"""
function loglikeprior(bnn::B, θ::Vector{T}, 
    x::Union{Vector{Matrix{T}}, Matrix{T}, Array{T, 3}}, 
    y::Union{Vector{T}, Matrix{T}}; num_batches = T(1)) where {B<:BNN, T}

    θnet, θhyper, θlike = split_params(bnn, θ)
    return num_batches*bnn.like(x, y, θnet, θlike) + bnn.prior(θnet, θhyper)
end

"""
Obtain the derivative of the unnormalised log posterior.
"""
function ∇loglikeprior(bnn::B, θ::Vector{T}, 
    x::Union{Vector{Matrix{T}}, Matrix{T}, Array{T, 3}}, 
    y::Union{Vector{T}, Matrix{T}}; num_batches = T(1)) where {B<:BNN, T}

    llp(θ) = loglikeprior(bnn, θ, x, y; num_batches = num_batches)
    v, g = Zygote.withgradient(llp, θ)

    return v, g[1]
end

function clip_gradient_value!(g, maxval=15)
    maxabs_g_val = maximum(abs.(g))
    if maxabs_g_val > maxval
        g .= maxval/maxabs_g_val .* g
    end
    return g
end

"""
    sample_prior_predictive(bnn::BNN, predict::Function, n::Int = 1;

Samples from the prior predictive. 

# Arguments
- `bnn` a BNN
- `predict` a function taking a network and returning a vector of predictions
- `n` number of samples

# Optional Arguments

- `rng` a RNG
"""
function sample_prior_predictive(bnn::BNN, predict::Function, n::Int = 1;
    rng::Random.AbstractRNG = Random.GLOBAL_RNG)

    prior = bnn.prior
    θnets = [sample_prior(prior, rng) for i=1:n]
    nets = [prior.nc(θ) for θ in θnets]
    ys = [predict(net) for net in nets]

    return ys
end

"""
    get_posterior_networks(bnn::BNN, ch::AbstractMatrix{T}) where {T}

Get the networks corresponding to posterior draws.

# Arguments
- `bnn` a BNN
- `ch` A Matrix of draws (columns are θ)
"""
function get_posterior_networks(bnn::BNN, ch::AbstractMatrix{T}) where {T}
    nets = [bnn.prior.nc(Float32.(split_params(bnn, ch[:, i])[1])) for i=1:size(ch, 2)]
    return nets
end

"""
    sample_posterior_predict(bnn::BNN, ch::AbstractMatrix{T}; x = bnn.x)

Sample from the posterior predictive distribution. 

# Arguments 

- `bnn`: a Bayesian Neural Network 
- `ch`: draws from the posterior. These should be either obtained using [`mcmc`](@ref) or [`bbb`](@ref) 
- `x`: explanatory variables. Default is to use the training data.

"""
function sample_posterior_predict(bnn::BNN, ch::AbstractMatrix{T}; x = bnn.x) where {T}
    θnets = [T.(split_params(bnn, ch[:, i])[1]) for i=1:size(ch, 2)]
    θlikes = [T.(split_params(bnn, ch[:, i])[3]) for i=1:size(ch, 2)]
    ys = reduce(hcat, [predict(bnn.like, x, θnet, θlike) for (θnet, θlike) in zip(θnets, θlikes)])
    return ys
end