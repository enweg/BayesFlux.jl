using Distributions

##########
# Normal #
##########

struct FeedforwardNormal{D<:Distributions.UnivariateDistribution}
    totparams::Int
    sigprior::D
    type::Type
end
function FeedforwardNormal(sigprior::D, type::Type) where {D<:Distributions.UnivariateDistribution}
    FeedforwardNormal(1, sigprior, type)
end

function loglike(FN::FeedforwardNormal, θ::AbstractVector, net::C, y::VecOrMat{T}, x::Matrix{T}) where {C<:Flux.Chain, T<:Real}
    yhat = vec(net(x))
    tsig = θ[1]
    sig = T(inverse(bijector(FN.sigprior))(tsig))
    
    return sum(logpdf.(Normal.(yhat, sig), y)) + logpdf(FN.sigprior, sig)
end
    

    
