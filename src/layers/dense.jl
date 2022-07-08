using Flux, Random, Distributions
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
