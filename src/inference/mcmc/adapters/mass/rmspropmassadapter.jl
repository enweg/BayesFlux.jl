"""
Use RMSProp as a preconditions/mass matrix adapter. This was proposed in 

Li, C., Chen, C., Carlson, D., & Carin, L. (2016, February). Preconditioned
stochastic gradient Langevin dynamics for deep neural networks. In Thirtieth
AAAI Conference on Artificial Intelligence for the use in SGLD and related
methods. 
"""
mutable struct RMSPropMassAdapter <: MassAdapter
    Minv::AbstractMatrix
    V::AbstractVector
    λ
    α 
    t::Int
    adapt_steps::Int
end
function RMSPropMassAdapter(adapt_steps = 1000; Minv::AbstractMatrix = Matrix(undef, 0, 0), 
    λ = 1f-5, α = 0.99f0)
    return RMSPropMassAdapter(Minv, Vector[], λ, α, 1, adapt_steps)
end
function (madapter::RMSPropMassAdapter)(s::MCMCState, θ::AbstractVector{T}, bnn::BNN, ∇θ) where {T}
    madapter.t > madapter.adapt_steps && return madapter.Minv
    madapter.t == 1 && size(madapter.Minv, 1) == 0 && (madapter.Minv = Diagonal(one.(θ)))
    madapter.t == 1 && (madapter.V = one.(θ))

    v, g = ∇θ(θ)
    g ./= norm(g)

    madapter.V = madapter.α * madapter.V + (1-madapter.α)*g.*g
    madapter.Minv = Diagonal(T(1) ./ (madapter.λ .+ sqrt.(madapter.V)))
    madapter.t += 1
    return madapter.Minv
end