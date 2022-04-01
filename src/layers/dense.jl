using Flux, Random, Distributions
using Optimisers
using Bijectors

###################
# Dense: standard #
###################

alphadist(::Val{Flux.Dense}, T::Type) = Gamma(T(1.0), T(1.0))

function retransform(::Val{Flux.Dense}, T::Type,  β::AbstractVector)
    # only transformed parameter is the first
    tα = β[1] 
    pα = alphadist(Val(Flux.Dense), T)
    # Fix type instability issue
    α = T(inverse(bijector(pα))(tα))
    return vcat(α, β[2:end])
end

function logprior(::Val{Flux.Dense}, T::Type, β::AbstractVector)
    # first one is hyper variance transformed by taking log 
    pα = alphadist(Val(Flux.Dense), T)
    β = retransform(Val(Flux.Dense), T, β)
    α = β[1]
    θ = β[2:end]

    return sum(logpdf.(Normal(zero(T), α), θ)) + logpdf(pα, α)
end

function sample(layer::Flux.Dense)
    θ, re = Flux.destructure(layer)
    T = eltype(θ)
    # Fixing type instability issue when sampling from Gamma
    α = T(rand(alphadist(Val(Flux.Dense), T)))
    θ = rand(Normal(zero(T), α), length(θ))
    return re(θ)
end

get_network_params(::Val{Flux.Dense}, θ::AbstractVector) = θ[2:end]

function BLayer(layer::Flux.Dense)
    θ, _ = Flux.destructure(layer)
    T = eltype(θ)
    totparams = length(θ) + 1 # one for the hyper variance prior
    lp(β) = logprior(Val(Flux.Dense), T, β)
    resamples(β) = retransform(Val(Flux.Dense), T, β)
    sampler() = sample(layer)

    return BLayer(layer, θ -> get_network_params(Val(Flux.Dense), θ), 
                  length(θ), totparams, lp, resamples, sampler)
end


