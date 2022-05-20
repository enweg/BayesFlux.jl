################################################################################
# Likelihood implementation are documented in implementation-likelihood.md
################################################################################

using Distributions
using Bijectors

################################################################################
# Normal 
################################################################################

struct FeedforwardNormal{T, F, D<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T, F}
    prior_σ::D
end
function FeedforwardNormal(nc::NetConstructor{T, F}, prior_σ::D) where {T, F, D<:Distribution}
    return FeedforwardNormal(1, nc, prior_σ)
end

function (l::FeedforwardNormal{T, F, D})(x::Matrix{T}, y::Vector{T}, θnet::Vector{T}, θlike::Vector{T}) where {T, F, D}
    net = l.nc(θnet)
    yhat = vec(net(x))
    tdist = transformed(l.prior_σ)
    sigma = invlink(l.prior_σ, θlike[1])
    n = length(y)

    # Using reparameterised likelihood 
    # Usually results in faster gradients
    return logpdf(MvNormal(zeros(n), I), (y-yhat)./sigma) - n*log(sigma) + logpdf(tdist, θlike[1])
end

function predict(l::FeedforwardNormal{T, F, D}, x::Matrix{T}, θnet::Vector{T}, θlike::Vector{T}) where {T, F, D}
    net = l.nc(θnet)
    yhat = vec(net(x))
    sigma = invlink(l.prior_σ, θlike[1])

    ypp = rand(MvNormal(yhat, sigma^2*I))
    return ypp
end

################################################################################
# T Distribution fixed df 
################################################################################

struct FeedforwardTDist{T, F, D<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T, F}
    prior_σ::D
    ν::T
end
function FeedforwardTDist(nc::NetConstructor{T, F}, prior_σ::D, ν::T) where {T, F, D}
    return FeedforwardTDist(1, nc, prior_σ, ν)
end

function (l::FeedforwardTDist{T, F, D})(x::Matrix{T}, y::Vector{T}, θnet::Vector{T}, θlike::Vector{T}) where {T, F, D}
    net = l.nc(θnet)
    yhat = vec(net(x))
    tdist = transformed(l.prior_σ)
    sigma = invlink(l.prior_σ, θlike[1])
    n = length(y)

    return sum(logpdf.(TDist(l.ν), (y-yhat)./sigma)) - n*log(sigma) + logpdf(tdist, θlike[1])
end

function predict(l::FeedforwardTDist{T, F, D}, x::Matrix{T}, θnet::Vector{T}, θlike::Vector{T}) where {T, F, D}
    net = l.nc(θnet)
    yhat = vec(net(x))
    sigma = invlink(l.prior_σ, θlike[1])
    n = length(yhat)

    ypp = sigma*rand(TDist(l.ν), n) + yhat
    return ypp
end
