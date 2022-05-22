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

function BNN(x, y, like, prior, init)
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

function loglikeprior(bnn::B, θ::Vector{T}, 
    x::Union{Vector{Matrix{T}}, Matrix{T}}, 
    y::Union{Vector{T}, Matrix{T}}; num_batches = T(1)) where {B<:BNN, T}

    θnet, θhyper, θlike = split_params(bnn, θ)
    return num_batches*bnn.like(x, y, θnet, θlike) + bnn.prior(θnet, θhyper)
end

function ∇loglikeprior(bnn::B, θ::Vector{T}, 
    x::Union{Vector{Matrix{T}}, Matrix{T}}, 
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
# function lprior(bnn::BNN, θ::AbstractVector)
#     θ = θ[bnn.loglikelihood.totparams+1:end]
#     T = bnn.type
#     logprior = zero(T)
#     s = 1
#     for bl in bnn.blayers 
#         totparams = bl.totparams
#         e = s + totparams - 1
#         β = view(θ, s:e)
#         s += totparams 
#         logprior += bl.lp(β)
#     end
    
#     return logprior
# end

# function lp(bnn::B, θ::AbstractVector) where {B<:BNN}
#     return lp(bnn, θ, bnn.x, bnn.y)
# end

# function lp(bnn::B, θ::AbstractVector, x::Union{Matrix{T}, Vector{Matrix{T}}}, y::Vector{T}) where {B<:BNN, T<:Real}
#     return lprior(bnn, θ) + loglike(bnn, bnn.loglikelihood, θ, y, x)
# end
