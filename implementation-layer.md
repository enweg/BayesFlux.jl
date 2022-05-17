# Layer Implementation

## Destruct

BFlux relies on the layers currently implemented in Flux. Thus, the first step in implementing a new layer for BFlux is to implement a new layer for Flux. Once that is done, one must also implement a destruct method. For example, for the Dense layer this has the following form 

```julia
function destruct(cell::Flux.Dense)
    @unpack weight, bias, σ = cell
    θ = vcat(vec(weight), vec(bias))
    function re(θ::AbstractVector)
        s = 1
        pweight = length(weight)
        new_weight = reshape(θ[s:s+pweight-1], size(weight))
        s += pweight
        pbias = length(bias)
        new_bias = reshape(θ[s:s+pbias-1], size(bias))
        return Flux.Dense(new_weight, new_bias, σ)
    end
    return θ, re
end
```

The destruct method takes as input a cell with the type of the cell being the newly implemented layer. It must return a vector containing all network parameter that should be trained/inferred and a function that given a vector of the right length can restructure the layer. **Note: Flux also implements a general destructure and restructure method. In my experience, this often caused problems in AD and thus until this is more stable, BFlux will stick with this manual setup**. 

When a user calls `BNN`, what is happening underneith is that we call `destruct` for each layer in the network. So if, like for recurrent networks, the network does not contain a pure cell, but rather a recurrent state of a cell, then one should implement the `destruct` method with regard to that recurrent cell. For an RNN this looks like the following: 

```julia
function destruct(cell::Flux.Recur{R}) where {R<:Flux.RNNCell}
    @unpack σ, Wi, Wh, b, state0 = cell.cell
    # θ = vcat(vec(Wi), vec(Wh), vec(b), vec(state0))
    θ = vcat(vec(Wi), vec(Wh), vec(b))
    function re(θ::AbstractVector)
        s = 1
        pWi = length(Wi)
        new_Wi = reshape(θ[s:s+pWi-1], size(Wi))
        s += pWi
        pWh = length(Wh)
        new_Wh = reshape(θ[s:s+pWh-1], size(Wh))
        s += pWh
        pb = length(b)
        new_b = reshape(θ[s:s+pb-1], size(b))
        s += pb
        # pstate0 = length(state0)
        # new_state0 = reshape(θ[s:s+pstate0-1], size(state0))
        new_state0 = zeros(size(state0)) 
        return Flux.Recur(Flux.RNNCell(σ, new_Wi, new_Wh, new_b, new_state0))
    end
    return θ, re
end
```

As can be seen from the commented out lines, we are currently not inferring the initial state. While this woule be great and could theoretically be done in a Bayesian setting, it also often seems to cause bad mixing and other difficulties in the inferential process. 

## Prior

Additionally to the `destruct` method, all layers should also implement a `logprior` method specifying the prior for all parameters of the cell. Currently BFlux is imposing a Gaussian prior for all parameters of a cell with a Gamma hyperprior on the standard deviation. Because BFlux needs parameters to take real values, we are transforming the Gamma prior using `Bijectors.jl`. Formally, for the cell parameters $\theta_i$ we are defining the following priors

$$
\theta_i \sim N(0, \alpha_i) \\
\alpha_i \sim Gamma(2,2)
$$

For a dense layer, the `logprior` method looks like this: 

```julia
alphadist(::Val{Flux.Dense}, T::Type) = Gamma(T(2.0), T(1.0))

function retransform(::Val{Flux.Dense}, T::Type,  β::AbstractVector)
    tα = β[1:Int(length(β)/2)] 
    pα = alphadist(Val(Flux.Dense), T)
    # Fix type instability issue
    α = T.(invlink.(pα, tα))
    return vcat(α, β[Int(length(β)/2)+1:end])
end

function logprior(::Val{Flux.Dense}, T::Type, β::AbstractVector)
    # all parameters have their own hyper prior on variance
    # This is the same hyperprior for all
    pα = alphadist(Val(Flux.Dense), T)
    β = retransform(Val(Flux.Dense), T, β)
    α = β[1:Int(length(β)/2)]
    θ = β[Int(length(β)/2)+1:end]

    # return sum(logpdf.(Normal.(zero(T), α), θ)) + sum(logpdf.(pα, α))
    return sum(logpdf.(Normal.(zero(T), α), θ)) + sum(logpdf_with_trans.(pα, α, true))
    # return sum(logpdf.(Normal.(zero(T), α), θ)) 
end
```

Note how we are also defining a `retransform` method which takes a vector and applies the needed transformations. 

### Defining a global prior 

One can also define a global prior for all parameters of a network. This works simply by overwriting the `lprior` function of BFlux. For example, if we do not like the default settings currently implemented and rather want a global prior on all the parameters in a network, then we can do the following: 

```julia
import BFlux: lprior

function lprior(bnn::BNN, θ)
    n = length(θ)
    return logpdf(MvNormal(zeros(n), ones(n)), θ)
end
```

This will overwrite all priors used by all layers of a network. 

## Sampling

Every layer should also implement a sampling method. This allows to sample from the prior distribution of networks. This is tightly coupled with the previous section of defining a prior, since we **must** use exactly the same distributions here. For a Dense layer, this looks like this: 

```julia
function sample(layer::Flux.Dense)
    θ, re = destruct(layer)
    T = eltype(θ)
    # Fixing type instability issue when sampling from Gamma
    α = T.(rand(alphadist(Val(Flux.Dense), T), length(θ)))
    θ = rand.(Normal.(zero(T), α))
    return re(θ)
end
```

## The Rest

After implementing all of the above functions, what is left are two smaller functions. The first `get_network_parameters` takes a vector and returns a vector containing only the network parameters. This is useful for cases where the prior uses hyperpriors and thus we are actually inferrring more than just the network parameters. If one uses a prior without hyperpriors, then this function should just return the input vector. Since we are using hyperpriors above, and since we are stacking the vector such that the hyperparameter (standard deviations of Gaussians) come first, we have the following implementation for a Dense layer: 

```julia
get_network_params(::Val{Flux.Dense}, θ::AbstractVector) = θ[Int(length(θ)/2)+1:end]
```

Lastly, we need to combine all the information above such that `BNN` can construct the right Bayesian version of the network. For this, `BLayer` needs to be implemented for each layer. Blayer takes a Flux or your implemented cell/layer, destructs it to obtain a vector representation of it, counts the total parameters that need to be estimated (for the Dense layer this is twice the network parameters due to hyperpriors), created a function for the `logprior` that only depends on a vector input and no longer on any other inputs, creates a function that given a sample containing possible transformed variables retransforms those into the original space, and defines how to sample from the layer. For a Dense layer, this looks like this: 

```julia
function BLayer(layer::Flux.Dense)
    θ, _ = destruct(layer)
    T = eltype(θ)
    totparams = length(θ)*2 # each parameter has a hyper parameter for the variance
    lp(β) = logprior(Val(Flux.Dense), T, β)
    resamples(β) = retransform(Val(Flux.Dense), T, β)
    sampler() = sample(layer)

    return BLayer(layer, θ -> get_network_params(Val(Flux.Dense), θ), 
                  length(θ), totparams, lp, resamples, sampler)
end
```