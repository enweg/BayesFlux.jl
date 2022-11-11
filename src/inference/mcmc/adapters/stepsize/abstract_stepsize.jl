
"""
Adapt the stepsize of MCMC algorithms.

# Implentation Details

## Mandatory Fields
- `l::Number` The stepsize. Will be used by the sampler.

## Mandatory Functions
- `(sadapter::StepsizeAdapter)(s::MCMCState, mh_probability::Number)` Every
  stepsize adapter must be callable with arguments, being the sampler itself and
  the Metropolis-Hastings acceptance probability. The method must return the new
  stepsize.
"""
abstract type StepsizeAdapter end

function (sadater::StepsizeAdapter)(s::MCMCState, mh_probability)
    error("$(typeof(sadapter)) is not callable. Please read the documentation of StepsizeAdapter.")
end
