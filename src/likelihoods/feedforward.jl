################################################################################
# Likelihood implementation are documented in implementation-likelihood.md
################################################################################

using Distributions

################################################################################
# Normal 
################################################################################

struct FeedforwardNormal{D<:Distributions.UnivariateDistribution}
    totparams::Int
    sigprior::D
    type::Type
end
function FeedforwardNormal(sigprior::D, type::Type) where {D<:Distributions.UnivariateDistribution}
    FeedforwardNormal(1, sigprior, type)
end

function loglike(bnn::BNN, FN::FeedforwardNormal{D}, θ::AbstractVector, y::VecOrMat{T}, x::Matrix{T}) where {C<:Flux.Chain, T<:Real, D<:Distributions.UnivariateDistribution}
    net_params = get_network_params(bnn, θ)
    net = bnn.re(net_params)
    θ = θ[1:FN.totparams]
    yhat = vec(net(x))
    tsig = θ[1]
    sig = T(invlink(FN.sigprior, tsig))
    
    return sum(logpdf.(Normal.(yhat, sig), y)) + logpdf_with_trans(FN.sigprior, sig, true)
end

function predict(bnn::BNN, FN::FeedforwardNormal{D}, draws::Matrix{T}; newx = bnn.x) where {D<:Distributions.UnivariateDistribution, T<:Real}
    netparams = [get_network_params(bnn, θ) for θ in eachcol(draws)]
    nethats = [bnn.re(np) for np in netparams]
    sigmas = draws[1,:]
    sigmas = invlink.([FN.sigprior], sigmas)
    yhats = [vec(nn(newx)) for nn in nethats]
    yhats = [rand.(Normal.(yh, sig)) for (yh, sig) in zip(yhats, sigmas)]
    return hcat(yhats...)
end

################################################################################
# T Distribution fixed df 
################################################################################

struct FeedforwardTDist{D<:Distributions.UnivariateDistribution, T}
    totparams::Int
    sigprior::D
    type::Type
    nu::T
end
function FeedforwardTDist(sigprior::D, nu::T) where {D<:Distributions.UnivariateDistribution, T<:Real}
    FeedforwardTDist(1, sigprior, T, nu)
end

function loglike(bnn::BNN, FT::FeedforwardTDist{D, R}, θ::AbstractVector, y::VecOrMat{T}, x::Matrix{T}) where {C<:Flux.Chain, T<:Real, D<:Distributions.UnivariateDistribution, R<:Real}
    net_params = get_network_params(bnn, θ)
    net = bnn.re(net_params)
    θ = θ[1:FT.totparams]
    yhat = vec(net(x))
    tsig = θ[1]
    sig = T(invlink(FT.sigprior, tsig))
    N = length(y)

    return sum(logpdf.(TDist(FT.nu), (y .- yhat)./sig)) - N*log(sig) + logpdf_with_trans(FT.sigprior, sig, true)
end

function predict(bnn::BNN, FT::FeedforwardTDist{D, R}, draws::Matrix{T}; newx = bnn.x) where {D<:Distributions.UnivariateDistribution, T<:Real, R<:Real}
    netparams = [get_network_params(bnn, θ) for θ in eachcol(draws)]
    nethats = [bnn.re(np) for np in netparams]
    sigmas = draws[1,:]
    sigmas = invlink.([FT.sigprior], sigmas)
    yhats = [vec(nn(newx)) for nn in nethats]
    yhats = [yh .+ sig*rand(TDist(FT.nu)) for (yh, sig) in zip(yhats, sigmas)]
    return hcat(yhats...)
end



    
