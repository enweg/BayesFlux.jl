

"""
 Adapt the mass matrix in MCMC and especially dynamic MCMCM methods such as 
 HMC, GGMC, SGLD, SGNHT, ...

 # Implementation Details

 ## Mandatory Fields
 - `Minv::AbstractMatrix`: The inverse mass matrix used in HMC, GGMC, ...

 ## Mandatory Functions
 - `(madapter::MassAdapter)(s::MCMCState, θ::AbstractVector, bnn, ∇θ)`: Every
   mass adapter must be callable and have the sampler state, the current sample,
   the BNN and a gradient function as arguments. It must return the new `Minv`
   Matrix.

"""
abstract type MassAdapter end