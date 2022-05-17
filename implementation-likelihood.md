# How to implement custom likelihoods? 

BFlux currently comes with a Gaussian and Student-t likelihood for both the feedforward and seq-to-one setting. If these likelihoods are not appropriate for the task at hand, then the user can implement a custom likelihood by defining the following four parts

1. A likelihood type. This should contain all the necessary information. 
2. A Constructor; This is not mandatory but often is helpful if the likelihood should have certain default values.
3. The `loglike` function which is being used for any mode finding or inferential method. 
4. The `predict` method which is being used to obtain posterior predictive draws. 

The following explains all four parts in more detail: 

## A Likelihood type

Every likelihood needs to be implemented as a type/struct. This struct should contain all information needed to calculate the likelihood, besides the actual data and paramters. Moreover, if the likelihood introduces any additional parameters that should be inferred and that are not part of any of the network parameters, so that no prior was defined for it yet, then the likelihood type must also contain the prior distribution of these additionally introduced parameters. All this likely becomes clearer when looking at how the Gaussian likelihood is implemented for the feedforward case: 

```julia
struct FeedforwardNormal{D<:Distributions.UnivariateDistribution}
    totparams::Int
    sigprior::D
    type::Type
end
function FeedforwardNormal(sigprior::D, type::Type) where {D<:Distributions.UnivariateDistribution}
    FeedforwardNormal(1, sigprior, type)
end
```

**Every likelihood type must hape a field `totparams` which contains the additional number of parameters introduced1**. In the Gaussian likelihood case, we also want to infer the standard deviation. Thus, we introduce an additional parameter. As such, we will later on set `totparams=1`. Addtitionally we need a field holding the prior distribution of this additional parameter. The `type` field can be ignored and will likely disappear in a future version of BFlux. 

We also create a constructor which makes our life easier. Since `totparams=1` always, we just set it to one and do not even take it as an argument for the constructor. Again, the type argument is a relict and will soon disappear. 

Compare the above to how the Student-t likelihood type for the feedforward setting looks like

```julia
struct FeedforwardTDist{D<:Distributions.UnivariateDistribution, T}
    totparams::Int
    sigprior::D
    type::Type
    nu::T
end
function FeedforwardTDist(sigprior::D, nu::T) where {D<:Distributions.UnivariateDistribution, T<:Real}
    FeedforwardTDist(1, sigprior, T, nu)
end
```
The Student-t distribution implemented in BFlux is a scaled one, and thus we again $\sigma$ as an additional parameter, together with its prior distribution. Note though that the degrees of freedom do not get a distribution but rather a fixed value. We currently opted for this due to the difficulty in inferring the degrees of freedom. Our constructor thus takes a prior distribution for $\sigma$ and a value for the degrees of freedom. 

The likelihoods for the seq-to-one setting are pretty much the same and thus will not be discussed here. Have a look at the code and if questions arise, send me an email. 

## The `loglike` function

After implementing the likelihood type, we need to implement the `loglike` function. For the feedforward Gaussian case, this has the following form: 

```julia
function loglike(bnn::BNN, FN::FeedforwardNormal{D}, θ::AbstractVector, y::VecOrMat{T}, x::Matrix{T}) where {C<:Flux.Chain, T<:Real, D<:Distributions.UnivariateDistribution}
    net_params = get_network_params(bnn, θ)
    net = bnn.re(net_params)
    θ = θ[1:FN.totparams]
    yhat = vec(net(x))
    tsig = θ[1]
    sig = T(invlink(FN.sigprior, tsig))
    
    return sum(logpdf.(Normal.(yhat, sig), y)) + logpdf_with_trans(FN.sigprior, sig, true)
end
```

In maths, we have the following

$$
y_i \sim N(net(x_i)_\theta, \sigma)
$$

Here the first line takes a vector representation of all parameters of the BNN (including hyper prior parameters), and obtains only those parameters corresponding to the actual network. We then use those to obtain the network which is then used to obtain `yhat`. We also obtain our transformed $\sigma$ which we need to transform back to the original space, again using `Bijectors.jl`. Having done all of this, we return the loglikelihood plus the logprior of all additionally introduced parameters (here the standard deviation; **note we are working in transformed space**). 

The Student-t case looks very similar. For the seq-to-one case we have the following for the Gaussian likelihood

```julia
function loglike(bnn::BNN, STON::SeqToOneNormal{D}, θ::AbstractVector, y::VecOrMat{T}, x::Vector{Matrix{T}}) where {C<:Flux.Chain, T<:Real, D<:Distributions.UnivariateDistribution}
    net_params = get_network_params(bnn, θ)
    net = bnn.re(net_params)
    θ = θ[1:STON.totparams]
    Flux.reset!(net)
    yhat = vec([net(xx) for xx in x][end])
    tsig = θ[1]
    sig = T(invlink(STON.sigprior, tsig))

    return sum(logpdf.(Normal.(yhat, sig), y)) + logpdf_with_trans(STON.sigprior, sig, true)
end
```

Note how the only true difference lies in how we calculate `yhat`.

## Prediction

To be able to obtain posterior predictive draws, it is not enough to create a network per sample, feed through the data and obtain an estimate. This would not completely take into account our uncertainty. Instead, we must create a network for each sample, use that network as the mean of our likelihood (this is usually the case) and add the additional uncertainty by drawing from the likelihood given this mean and all additional samples corresponding to the addtional parameters of the likelihood. The make this clearer, here is what we do for the Gaussian feedforward case, which is essentially what we also do for all others: 

```julia
function predict(bnn::BNN, FN::FeedforwardNormal{D}, draws::Matrix{T}; newx = bnn.x) where {D<:Distributions.UnivariateDistribution, T<:Real}
    netparams = [get_network_params(bnn, θ) for θ in eachcol(draws)]
    nethats = [bnn.re(np) for np in netparams]
    sigmas = draws[1,:]
    sigmas = invlink.([FN.sigprior], sigmas)
    yhats = [vec(nn(newx)) for nn in nethats]
    yhats = [rand.(Normal.(yh, sig)) for (yh, sig) in zip(yhats, sigmas)]
    return hcat(yhats...)
end
```

**Every predict method must have the signature above with types being appropriately changed**

That's it. If you read through how to implement a layer and thus a prior for layers and this document, then you shoudl be able to adjust BFlux to your needs. If there is still something unclear please let me know. BFlux is still in its infancy and thus a lot of things are still awkward and not the fastest or most efficient yet. I hope this will be sorted out over time. 