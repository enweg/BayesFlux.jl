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
    function re(θ::Vector{T}) where {T}
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
        new_state0 = zeros(T, size(state0))
        return Flux.Recur(Flux.RNNCell(σ, new_Wi, new_Wh, new_b, new_state0))
    end
    return θ, re
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
    function re(θ::Vector{T}) where {T}
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
        new_state01 = T.(state0[1])
        # s += pstate01 
        # pstate02 = length(state0[2])
        # new_state02 = reshape(θ[s:s+pstate02-1], size(state0[2]))
        new_state02 = T.(state0[2])
        return Flux.Recur(Flux.LSTMCell(new_Wi, new_Wh, new_b, (new_state01, new_state02)))
    end
    return θ, re
end