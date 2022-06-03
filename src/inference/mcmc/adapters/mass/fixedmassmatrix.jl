using LinearAlgebra
"""
Use a fixed inverse mass matrix.
"""
struct FixedMassAdapter <: MassAdapter 
    Minv::AbstractMatrix
end
FixedMassAdapter() = FixedMassAdapter(Matrix(undef, 0, 0))
function (madapter::FixedMassAdapter)(s::MCMCState, θ::AbstractMatrix{T}, bnn, ∇θ) where {T}
    size(madapter.Minv, 1) == 0 && (madapter.Minv = Diagonal(one.(θ)))
    return madapter.Minv
end