################################################################################
# Likelihood implementation are documented in implementation-likelihood.md
################################################################################
using Distributions
using LinearAlgebra

###############################################################################
# Sequence to one Normal 
################################################################################

struct SeqToOneNormal{D<:Distributions.UnivariateDistribution}
    totparams::Int
    sigprior::D
    type::Type
end
function SeqToOneNormal(sigprior::D, type::Type) where {D<:Distributions.UnivariateDistribution}
    SeqToOneNormal(1, sigprior, type)
end

function loglike(bnn::BNN, STON::SeqToOneNormal{D}, θ::AbstractVector, y::VecOrMat{T}, x::Vector{Matrix{T}}) where {C<:Flux.Chain, T<:Real, D<:Distributions.UnivariateDistribution}
    net_params = get_network_params(bnn, θ)
    net = bnn.re(net_params)
    θ = θ[1:STON.totparams]
    Flux.reset!(net)
    yhat = vec([net(xx) for xx in x][end])
    tsig = θ[1]
    sig = T(invlink(STON.sigprior, tsig))

    return sum(logpdf.(Normal.(yhat, sig), y)) + logpdf_with_trans(STON.sigprior, sig, true)
end

function predict(bnn::BNN, STON::SeqToOneNormal{D}, draws::Matrix{T}; newx = bnn.x) where {D<:Distributions.UnivariateDistribution, T<:Real}
    netparams = [get_network_params(bnn, θ) for θ in eachcol(draws)]
    nethats = [bnn.re(np) for np in netparams]
    sigmas = draws[1,:]
    sigmas = invlink.([STON.sigprior], sigmas)
    yhats = [vec([nn(xx) for xx in newx][end]) for nn in nethats]
    yhats = [yh .+ sig*randn(length(yh)) for (yh, sig) in zip(yhats, sigmas)]
    return hcat(yhats...)
end

################################################################################
# Sequence to one TDIST
################################################################################


struct SeqToOneTDist{D<:Distributions.UnivariateDistribution, T}
    totparams::Int
    sigprior::D
    type::Type
    nu::T
end
function SeqToOneTDist(sigprior::D, nu::T) where {D<:Distributions.UnivariateDistribution, T<:Real}
    SeqToOneTDist(1, sigprior, T, nu)
end

function loglike(bnn::BNN, STOT::SeqToOneTDist{D, R}, θ::AbstractVector, y::VecOrMat{T}, x::Vector{Matrix{T}}) where {C<:Flux.Chain, T<:Real, D<:Distributions.UnivariateDistribution, R<:Real}
    net_params = get_network_params(bnn, θ)
    net = bnn.re(net_params)
    θ = θ[1:STOT.totparams]
    Flux.reset!(net)
    yhat = vec([net(xx) for xx in x][end])
    tsig = θ[1]
    sig = T(invlink(STOT.sigprior, tsig))
    N = length(y)

    return sum(logpdf.(TDist(STOT.nu), (y .- yhat)./sig)) - N*log(sig) + logpdf_with_trans(STOT.sigprior, sig, true)
end

function predict(bnn::BNN, STOT::SeqToOneTDist{D, R}, draws::Matrix{T}; newx = bnn.x) where {D<:Distributions.UnivariateDistribution, T<:Real, R<:Real}
    netparams = [get_network_params(bnn, θ) for θ in eachcol(draws)]
    nethats = [bnn.re(np) for np in netparams]
    sigmas = draws[1,:]
    sigmas = invlink.([STOT.sigprior], sigmas)
    yhats = [vec([nn(xx) for xx in newx][end]) for nn in nethats]
    yhats = [yh .+ sig*rand(TDist(STOT.nu)) for (yh, sig) in zip(yhats, sigmas)]
    return hcat(yhats...)
end