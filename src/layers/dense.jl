using Flux, Random, Distributions
using Optimisers
using Bijectors
using Parameters

################################################################################
# Dense: standard 
################################################################################

function destruct(cell::Flux.Dense)
    @unpack weight, bias, σ = cell
    θ = vcat(vec(weight), vec(bias))
    function re(θ::AbstractVector)
        s = 1
        pweight = length(weight)
        new_weight = reshape(θ[s:s+pweight-1], size(weight))
        s += pweight
        pbias = length(bias)
        new_bias = reshape(θ[s:s+pbias-1], size(bias))
        return Flux.Dense(new_weight, new_bias, σ)
    end
    return θ, re
end

alphadist(::Val{Flux.Dense}, T::Type) = Gamma(T(2.0), T(1.0))

function retransform(::Val{Flux.Dense}, T::Type,  β::AbstractVector)
    tα = β[1:Int(length(β)/2)] 
    pα = alphadist(Val(Flux.Dense), T)
    # Fix type instability issue
    # α = T.(inverse(bijector(pα)).(tα))
    α = T.(invlink.(pα, tα))
    return vcat(α, β[Int(length(β)/2)+1:end])
end

function logprior(::Val{Flux.Dense}, T::Type, β::AbstractVector)
    # all parameters have their own hyper prior on variance
    # This is the same hyperprior for all
    pα = alphadist(Val(Flux.Dense), T)
    β = retransform(Val(Flux.Dense), T, β)
    α = β[1:Int(length(β)/2)]
    θ = β[Int(length(β)/2)+1:end]

    # return sum(logpdf.(Normal.(zero(T), α), θ)) + sum(logpdf.(pα, α))
    return sum(logpdf.(Normal.(zero(T), α), θ)) + sum(logpdf_with_trans.(pα, α, true))
    # return sum(logpdf.(Normal.(zero(T), α), θ)) 
end

function sample(layer::Flux.Dense)
    θ, re = destruct(layer)
    T = eltype(θ)
    # Fixing type instability issue when sampling from Gamma
    α = T.(rand(alphadist(Val(Flux.Dense), T), length(θ)))
    θ = rand.(Normal.(zero(T), α))
    return re(θ)
end

get_network_params(::Val{Flux.Dense}, θ::AbstractVector) = θ[Int(length(θ)/2)+1:end]

function BLayer(layer::Flux.Dense)
    θ, _ = destruct(layer)
    T = eltype(θ)
    totparams = length(θ)*2 # each parameter has a hyper parameter for the variance
    lp(β) = logprior(Val(Flux.Dense), T, β)
    resamples(β) = retransform(Val(Flux.Dense), T, β)
    sampler() = sample(layer)

    return BLayer(layer, θ -> get_network_params(Val(Flux.Dense), θ), 
                  length(θ), totparams, lp, resamples, sampler)
end


