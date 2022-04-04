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
    
