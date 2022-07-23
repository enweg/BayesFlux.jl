

"""
    InitialiseAllSame(dist::D, like::BNNLikelihood, prior::NetworkPrior)

Initialise all values by drawing from `dist`
"""
struct InitialiseAllSame{D<:Distribution} <: BNNInitialiser
    length_θnet::Int
    length_hyper::Int
    length_like::Int
    type::Type
    dist::D
end
function InitialiseAllSame(dist::D, like::BNNLikelihood, prior::NetworkPrior) where {D}
    length_θnet = like.nc.num_params_network
    length_hyper = prior.num_params_hyper
    length_like = like.num_params_like
    type = eltype(like.nc.θ)
    return InitialiseAllSame(length_θnet, length_hyper, length_like, type, dist)
end
function (init::InitialiseAllSame{D})(rng::AbstractRNG = Random.GLOBAL_RNG) where {D}
    θnet = rand(rng, init.dist, init.length_θnet)
    θhyper = rand(rng, init.dist, init.length_hyper)
    θlike = rand(rng, init.dist, init.length_like) 
    return init.type.(θnet), init.type.(θhyper), init.type.(θlike)
end

