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

function loglike(STON::SeqToOneNormal{D}, θ::AbstractVector, net::C, y::VecOrMat{T}, x::Vector{Matrix{T}}) where {C<:Flux.Chain, T<:Real, D<:Distributions.UnivariateDistribution}
    Flux.reset!(net)
    yhat = vec([net(xx) for xx in x][end])
    tsig = θ[1]
    sig = T(inverse(bijector(STON.sigprior))(tsig))

    return sum(logpdf.(Normal.(yhat, sig), y)) + logpdf_with_trans(STON.sigprior, sig, true)
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

function loglike(FT::SeqToOneTDist{D, R}, θ::AbstractVector, net::C, y::VecOrMat{T}, x::Vector{Matrix{T}}) where {C<:Flux.Chain, T<:Real, D<:Distributions.UnivariateDistribution, R<:Real}
    Flux.reset!(net)
    yhat = vec([net(xx) for xx in x][end])
    tsig = θ[1]
    sig = T(inverse(bijector(FT.sigprior))(tsig))
    N = length(y)

    return sum(logpdf.(TDist(FT.nu), (y .- yhat)./sig)) - N*log(sig) + logpdf_with_trans(FT.sigprior, sig, true)
end
