
"""
    abstract type BNNInitialiser end

To initialise BNNs, BNNInitialisers are used. These must be callable and must
return three vectors 

- `θnet` the initial values for the network parameters
- `θhyper` the initial values for any hyperparameters introduced by the
  NetworkPrior
- `θlike` the initial values for any extra parameters introduced by the
  likelihood 

# Implementation

Every BNNInitialiser must be callable and must return the above three things: 

    (init::BNNInitialiser)(rng::AbstractRNG) -> (θnet, θhyper, θlike)

"""
abstract type BNNInitialiser end

(init::BNNInitialiser)() = error("The initialiser is not properly implemented. Please see the BNNInitialiser documentation.")