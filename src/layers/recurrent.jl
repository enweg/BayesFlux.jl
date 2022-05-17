################################################################################
# Layer implementations are documented in implementation-layer.md 
################################################################################
using Flux, Random, Distributions
using Optimisers
using Bijectors
using Parameters

################################################################################
# RNN
################################################################################

function destruct(cell::Flux.Recur{R}) where {R<:Flux.RNNCell}
    @unpack σ, Wi, Wh, b, state0 = cell.cell
    # θ = vcat(vec(Wi), vec(Wh), vec(b), vec(state0))
    θ = vcat(vec(Wi), vec(Wh), vec(b))
    function re(θ::AbstractVector)
        s = 1
        pWi = length(Wi)
        new_Wi = reshape(θ[s:s+pWi-1], size(Wi))
        s += pWi
        pWh = length(Wh)
        new_Wh = reshape(θ[s:s+pWh-1], size(Wh))
        s += pWh
        pb = length(b)
        new_b = reshape(θ[s:s+pb-1], size(b))
        s += pb
        # pstate0 = length(state0)
        # new_state0 = reshape(θ[s:s+pstate0-1], size(state0))
        new_state0 = zeros(size(state0)) 
        return Flux.Recur(Flux.RNNCell(σ, new_Wi, new_Wh, new_b, new_state0))
    end
    return θ, re
end

alphadist(::Val{Flux.RNN}, T::Type) = Gamma(T(1.0), T(1.0))

function retransform(::Val{Flux.RNN}, T::Type,  β::AbstractVector)
    tα = β[1:Int(length(β)/2)] 
    pα = alphadist(Val(Flux.RNN), T)
    # Fix type instability issue
    α = T.(invlink(pα, tα))
    return vcat(α, β[Int(length(β)/2)+1:end])
end

function logprior(::Val{Flux.RNN}, T::Type, β::AbstractVector)
    # all parameters have their own hyper prior on variance
    # This is the same hyperprior for all
    pα = alphadist(Val(Flux.RNN), T)
    β = retransform(Val(Flux.RNN), T, β)
    α = β[1:Int(length(β)/2)]
    θ = β[Int(length(β)/2)+1:end]

    return sum(logpdf.(Normal.(zero(T), α), θ)) + sum(logpdf_with_trans.(pα, α, true))
end

function sample(layer::Flux.Recur{R}) where {R<:Flux.RNNCell}
    θ, re = destruct(layer)
    T = eltype(θ)
    # Fixing type instability issue when sampling from Gamma
    α = T.(rand(alphadist(Val(Flux.RNN), T), length(θ)))
    θ = rand.(Normal.(zero(T), α))
    return re(θ)
end

get_network_params(::Val{Flux.RNN}, θ::AbstractVector) = θ[Int(length(θ)/2)+1:end]

function BLayer(layer::Flux.Recur{R}) where {R<:Flux.RNNCell}
    θ, _ = destruct(layer)
    T = eltype(θ)
    totparams = length(θ)*2 # each parameter has a hyper parameter for the variance
    lp(β) = logprior(Val(Flux.RNN), T, β)
    resamples(β) = retransform(Val(Flux.RNN), T, β)
    sampler() = sample(layer)

    return BLayer(layer, θ -> get_network_params(Val(Flux.RNN), θ), 
                  length(θ), totparams, lp, resamples, sampler)
end

################################################################################
# LSTM
################################################################################

function destruct(cell::Flux.Recur{R}) where {R<:Flux.LSTMCell}
    @unpack Wi, Wh, b, state0 = cell.cell
    # state 0 has two states, one for h and one for c
    # see wikipedia article
    # θ = vcat(vec(Wi), vec(Wh), vec(b), vec(state0[1]), vec(state0[2]))
    θ = vcat(vec(Wi), vec(Wh), vec(b))
    function re(θ::AbstractVector)
        s = 1
        pWi = length(Wi)
        new_Wi = reshape(θ[s:s+pWi-1], size(Wi))
        s += pWi
        pWh = length(Wh)
        new_Wh = reshape(θ[s:s+pWh-1], size(Wh))
        s += pWh
        pb = length(b)
        new_b = reshape(θ[s:s+pb-1], size(b))
        # s += pb 
        # pstate01 = length(state0[1])
        # new_state01 = reshape(θ[s:s+pstate01-1], size(state0[1]))
        new_state01 = Float64.(state0[1])
        # s += pstate01 
        # pstate02 = length(state0[2])
        # new_state02 = reshape(θ[s:s+pstate02-1], size(state0[2]))
        new_state02 = Float64.(state0[2])
        return Flux.Recur(Flux.LSTMCell(new_Wi, new_Wh, new_b, (new_state01, new_state02)))
    end
    return θ, re
end


alphadist(::Val{Flux.LSTM}, T::Type) = Gamma(T(1.0), T(1.0))

function retransform(::Val{Flux.LSTM}, T::Type,  β::AbstractVector)
    tα = β[1:Int(length(β)/2)] 
    pα = alphadist(Val(Flux.LSTM), T)
    # Fix type instability issue
    α = T.(invlink(pα, tα))
    return vcat(α, β[Int(length(β)/2)+1:end])
end

function logprior(::Val{Flux.LSTM}, T::Type, β::AbstractVector)
    # all parameters have their own hyper prior on variance
    # This is the same hyperprior for all
    pα = alphadist(Val(Flux.LSTM), T)
    β = retransform(Val(Flux.LSTM), T, β)
    α = β[1:Int(length(β)/2)]
    θ = β[Int(length(β)/2)+1:end]

    return sum(logpdf.(Normal.(zero(T), α), θ)) + sum(logpdf_with_trans.(pα, α, true))
end

function sample(layer::Flux.Recur{R}) where {R<:Flux.LSTMCell}
    θ, re = destruct(layer)
    T = eltype(θ)
    # Fixing type instability issue when sampling from Gamma
    α = T.(rand(alphadist(Val(Flux.LSTM), T), length(θ)))
    θ = rand.(Normal.(zero(T), α))
    return re(θ)
end

get_network_params(::Val{Flux.LSTM}, θ::AbstractVector) = θ[Int(length(θ)/2)+1:end]

function BLayer(layer::Flux.Recur{R}) where {R<:Flux.LSTMCell}
    θ, _ = destruct(layer)
    T = eltype(θ)
    totparams = length(θ)*2 # each parameter has a hyper parameter for the variance
    lp(β) = logprior(Val(Flux.LSTM), T, β)
    resamples(β) = retransform(Val(Flux.LSTM), T, β)
    sampler() = sample(layer)

    return BLayer(layer, θ -> get_network_params(Val(Flux.LSTM), θ), 
                  length(θ), totparams, lp, resamples, sampler)
end