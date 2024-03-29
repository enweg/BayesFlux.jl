"""
Use the variances as the diagonal of the inverse mass matrix as used in HMC, 
GGMC, ...; 

# Fields
- `Minv`: Inverse mass matrix as used in HMC, SGLD, GGMC, ...
- `adapt_steps`: Number of adaptation steps. 
- `windowlength`: Lookback length for calculation of covariance.
- `t`: Current step. 
- `kappa`: How much to shrink towards the identity.
- `epsilon`: Small value to add to diagonal to avoid numerical instability.
"""
mutable struct DiagCovMassAdapter{T} <: MassAdapter
    Minv::AbstractMatrix{T}
    adapt_steps::Int
    windowlength::Int
    t::Int
    kappa::T
    epsilon::T
end

function DiagCovMassAdapter(
    adapt_steps::Int,
    windowlength::Int;
    Minv::AbstractMatrix=Matrix(undef, 0, 0), 
    kappa::T=0.5f0, 
    epsilon=1.0f-6
) where {T}

    size(Minv, 1) == 0 && (Minv = Matrix{T}(undef, 0, 0))

    return DiagCovMassAdapter(Minv, adapt_steps, windowlength, 1, kappa, epsilon)
end

function (madapter::DiagCovMassAdapter{T})(s::MCMCState, 
    θ::AbstractVector{T}, 
    bnn::BNN, 
    ∇θ
) where {T}

    madapter.t == 1 && size(madapter.Minv, 1) == 0 && (madapter.Minv = Diagonal(one.(θ)))
    madapter.t > madapter.adapt_steps && return madapter.Minv
    madapter.windowlength > s.nsampled && return madapter.Minv

    madapter.Minv = (T(1) - madapter.kappa) * Diagonal(vec(var(permutedims(s.samples[:, s.nsampled-madapter.windowlength+1:s.nsampled]); dims=1))) + madapter.kappa * I
    madapter.Minv = madapter.Minv + madapter.epsilon * I
    madapter.t += 1

    return madapter.Minv
end