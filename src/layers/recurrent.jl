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
