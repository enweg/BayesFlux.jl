
"""
    abstract type BNNLikelihood end

Every likelihood must be a subtype of BNNLikelihood and must implement at least
the following fields: 

- `num_params_like`: The number of additional parameters introduced for which
  inference will be done (i.e. σ for a Gaussian but not ν for a T-Dist if df is
  not inferred)
- `nc`: A NetConstructor

Every BNNLikelihood must be callable in the following way 

    (l::BNNLikelihood)(x, y, θnet, θlike)

- `x` either all data or a minibatch
- `y` either all data or a minibatch
- `θnet` are the network parameter
- `θlike` are the likelihood parameters. If no additional parameters were
  introduced, this will be an empty array

Every BNNLikelihood must also implement a predict method which should draw from
the posterior predictive given network parameters and likelihood parameters. 

    predict(l::BNNLikelihood, x, θnet, θlike)

- `l` the BNNLikelihood
- `x` the input data 
- `θnet` network parameters
- `θlike` likelihood parameters

"""
abstract type BNNLikelihood end

function (l::BNNLikelihood)(x::Union{Vector{Matrix{T}}, Matrix{T}}, 
  y::Union{Vector{T}, Matrix{T}}, 
  θnet::AbstractVector, θlike::AbstractVector) where {T}

  error("Seems like your likelihood is not callable. Please see the documentation for BNNLikelihood.")
end

predict(l::BNNLikelihood, x, θnet, θlike) = error("Seems like your likelihood did not implement a predict method. Please see the documentation for BNNLikelihood.")
