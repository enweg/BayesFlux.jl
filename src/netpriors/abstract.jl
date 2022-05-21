
"""
    abstract type NetworkPrior end

All priors over the network parameters are subtypes of NetworkPrior. 

# Required Fields

- `num_params_hyper`: Number of hyper priors (only those that should be
  inferred)
- `nc::NetConstructor`: NetConstructor

# Required implementations

Each NetworkPrior must be callable in the form of 

    (np::NetworkPrior)(θnet, θhyper)

- `θnet` is a vector of network parameters
- `θhyper` is a vector of hyper parameters

The return value must be the logprior including all hyper-priors. 

Each NetworkPrior must also implement a sample function returning a vector
`θnet` of network parameters drawn from the prior. 

    sample_prior(np::NetworkPrior)(rng::AbstractRNG)


"""
abstract type NetworkPrior end

(np::NetworkPrior)(θnet, θhyper) = error("Seems like your network prior is not implemented correctly. Please consult the documentation for NetworkPrior.")

sample_prior(np::NetworkPrior, rng::Random.AbstractRNG) = error("Seems like your network prior is not implemented correctly. Please consult the documentation for NetworkPrior.")
