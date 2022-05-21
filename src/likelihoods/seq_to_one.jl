################################################################################
# Likelihood implementation are documented in implementation-likelihood.md
################################################################################
using Distributions
using LinearAlgebra

###############################################################################
# Sequence to one Normal 
################################################################################


struct SeqToOneNormal{T, F, D<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T, F}
    prior_σ::D
end
function SeqToOneNormal(nc::NetConstructor{T, F}, prior_σ::D) where {T, F, D<:Distribution}
    return SeqToOneNormal(1, nc, prior_σ)
end

function (l::SeqToOneNormal{T, F, D})(x::Vector{Matrix{T}}, y::Vector{T}, θnet::AbstractVector, θlike::AbstractVector) where {T, F, D}
    θnet = T.(θnet)
    θlike = T.(θlike)

    net = l.nc(θnet)
    yhat = vec([net(xx) for xx in x][end])
    tdist = transformed(l.prior_σ)
    sigma = invlink(l.prior_σ, θlike[1])
    n = length(y)

    # Using reparameterised likelihood 
    # Usually results in faster gradients
    return logpdf(MvNormal(zeros(n), I), (y-yhat)./sigma) - n*log(sigma) + logpdf(tdist, θlike[1])
end

function predict(l::SeqToOneNormal{T, F, D}, x::Vector{Matrix{T}}, θnet::AbstractVector, θlike::AbstractVector) where {T, F, D}

    θnet = T.(θnet)
    θlike = T.(θlike)

    net = l.nc(θnet)
    yhat = vec([net(xx) for xx in x][end])
    sigma = invlink(l.prior_σ, θlike[1])

    ypp = rand(MvNormal(yhat, sigma^2*I))
    return ypp
end


################################################################################
# Sequence to one TDIST
################################################################################

struct SeqToOneTDist{T, F, D<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T, F}
    prior_σ::D
    ν::T
end
function SeqToOneTDist(nc::NetConstructor{T, F}, prior_σ::D, ν::T) where {T, F, D}
    return SeqToOneTDist(1, nc, prior_σ, ν)
end

function (l::SeqToOneTDist{T, F, D})(x::Vector{Matrix{T}}, y::Vector{T}, θnet::AbstractVector, θlike::AbstractVector) where {T, F, D}
    θnet = T.(θnet)
    θlike = T.(θlike)

    net = l.nc(θnet)
    yhat = vec([net(xx) for xx in x][end])
    tdist = transformed(l.prior_σ)
    sigma = invlink(l.prior_σ, θlike[1])
    n = length(y)

    return sum(logpdf.(TDist(l.ν), (y-yhat)./sigma)) - n*log(sigma) + logpdf(tdist, θlike[1])
end

function predict(l::SeqToOneTDist{T, F, D}, x::Vector{Matrix{T}}, θnet::AbstractVector, θlike::AbstractVector) where {T, F, D}
    θnet = T.(θnet)
    θlike = T.(θlike)


    net = l.nc(θnet)
    yhat = vec([net(xx) for xx in x][end])
    sigma = invlink(l.prior_σ, θlike[1])
    n = length(yhat)

    ypp = sigma*rand(TDist(l.ν), n) + yhat
    return ypp
end