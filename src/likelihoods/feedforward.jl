using Distributions
using Bijectors

################################################################################
# Normal 
################################################################################

"""
    FeedforwardNormal(nc::NetConstructor{T, F}, prior_σ::D) where {T, F, D<:Distribution}

Use a Gaussian/Normal likelihood for a Feedforward architecture with a single output.

Assumes is a single output. Thus, the last layer must have output size one. 

# Arguments

- `nc`: obtained using [`destruct`](@ref)
- `prior_σ`: a prior distribution for the standard deviation

"""
struct FeedforwardNormal{T,F,D<:Distributions.Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
    prior_σ::D
end
function FeedforwardNormal(nc::NetConstructor{T,F}, prior_σ::D) where {T,F,D<:Distributions.Distribution}
    return FeedforwardNormal(1, nc, prior_σ)
end

function (l::FeedforwardNormal{T,F,D})(x::Matrix{T}, y::Vector{T}, θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)

    net = l.nc(θnet)
    yhat = vec(net(x))
    tdist = transformed(l.prior_σ)
    sigma = invlink(l.prior_σ, θlike[1])
    n = length(y)

    # Using reparameterised likelihood 
    # Usually results in faster gradients
    return logpdf(MvNormal(zeros(n), I), (y - yhat) ./ sigma) - n * log(sigma) + logpdf(tdist, θlike[1])
end

function predict(l::FeedforwardNormal{T,F,D}, x::Matrix{T}, θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)

    net = l.nc(θnet)
    yhat = vec(net(x))
    sigma = invlink(l.prior_σ, θlike[1])

    ypp = rand(MvNormal(yhat, sigma^2 * I))
    return ypp
end

################################################################################
# T Distribution fixed df 
################################################################################

"""
    FeedforwardTDist(nc::NetConstructor{T, F}, prior_σ::D, ν::T) where {T, F, D}

Use a Student-T likelihood for a Feedforward architecture with a single output
and known degress of freedom.

Assumes is a single output. Thus, the last layer must have output size one. 

# Arguments

- `nc`: obtained using [`destruct`](@ref)
- `prior_σ`: a prior distribution for the standard deviation
- `ν`: degrees of freedom

"""
struct FeedforwardTDist{T,F,D<:Distributions.Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
    prior_σ::D
    ν::T
end
function FeedforwardTDist(nc::NetConstructor{T,F}, prior_σ::D, ν::T) where {T,F,D}
    return FeedforwardTDist(1, nc, prior_σ, ν)
end

function (l::FeedforwardTDist{T,F,D})(x::Matrix{T}, y::Vector{T}, θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)

    net = l.nc(θnet)
    yhat = vec(net(x))
    tdist = transformed(l.prior_σ)
    sigma = invlink(l.prior_σ, θlike[1])
    n = length(y)

    return sum(logpdf.(TDist(l.ν), (y - yhat) ./ sigma)) - n * log(sigma) + logpdf(tdist, θlike[1])
end

function predict(l::FeedforwardTDist{T,F,D}, x::Matrix{T}, θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)

    net = l.nc(θnet)
    yhat = vec(net(x))
    sigma = invlink(l.prior_σ, θlike[1])
    n = length(yhat)

    ypp = sigma * rand(TDist(l.ν), n) + yhat
    return ypp
end
