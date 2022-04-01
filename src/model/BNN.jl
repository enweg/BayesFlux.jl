using Flux, Distributions, Random
using Zygote

struct BLayer{U, NP, F, V, S}
    unit::U
    gnp::NP
    nparams::Int # number parameters of network
    totparams::Int # number all parameters including hyperpriors
    lp::F
    resamples::V
    sample::S
end

struct BNN{C<:Flux.Chain, R, B<:BLayer, F, Ty, Tx} 
    net::C
    re::R
    type::Type
    nparams::Int # number parameters of network
    totparams::Int # number all parameters including hyper priors
    blayers::Vector{B}
    loglikelihood::F
    y::Ty
    x::Tx
end

function BNN(net::C, loglike::F, y::Ty, x::Tx) where {C<:Flux.Chain, F, Ty, Tx}
    θ, re = Flux.destructure(net)
    blayers = [BLayer(layer) for layer in net]
    totparams = sum([bl.totparams for bl in blayers]) + loglike.totparams
    nparams = sum([bl.nparams for bl in blayers])
    type = eltype(θ)
    return BNN(net, re, type, nparams, totparams, blayers, loglike, y, x)
end

function get_network_params(bnn::B, θ::AbstractVector) where {B<:BNN}
    θ = θ[bnn.loglikelihood.totparams+1:end]

    network_params = Zygote.Buffer(θ, bnn.nparams)
    s1 = 1
    s2 = 1
    for bl in bnn.blayers
        e = s2 + bl.totparams - 1
        β = view(θ, s2:e)
        s2 += bl.totparams
        np = bl.gnp(β)
        e1 = s1 + length(np) - 1
        network_params[s1:e1] = np
        s1 += length(np)
    end
    return copy(network_params)
end


function lp(bnn::B, θ::AbstractVector) where {B<:BNN}
    T = bnn.type 
    net_parameters = get_network_params(bnn, θ)
    net = bnn.re(net_parameters)

    loglike_θ = zero(T) 
    if (bnn.loglikelihood.totparams > 0)
        loglike_θ = θ[1:bnn.loglikelihood.totparams]
        θ = θ[bnn.loglikelihood.totparams+1:end]
    end
    logprior = zero(T)

    s = 1
    for bl in bnn.blayers 
        totparams = bl.totparams
        e = s + totparams - 1
        β = view(θ, s:e)
        s += totparams 
        logprior += bl.lp(β)
    end

    return logprior + loglike(bnn.loglikelihood, loglike_θ, net, bnn.y, bnn.x)
end

# function reconstruct_sample(bnn::B, θ::AbstractVector) where {B<:BNN}
#     s = 1
#     θ_rec = similar(θ)
#     for bl in bnn.blayers
#         nparams = bl.nparams 
#         e = s + nparams - 1
#         θ_rec[s:e] .= bl.resamples(view(θ, s:e))
#         s += nparams
#     end
#     return θ_rec
# end