################################################################################
# ADVI for BFlux
# FIXME: Something goes wrong using ADVI and empty vectors (θhyper)
################################################################################
using AdvancedVI, DistributionsAD
using ProgressMeter

"""
    advi(bnn::BNN, samples_per_step::Int, maxiters::Int, args...; kwargs...)

Estimate a BNN using Automatic Differentiation Variational Inference (ADVI). 

# Uses
- Uses ADVI as implemented in AdvancedVI.jl

# Arguments 
- `bnn`: A Bayesian Neural Network
- `samples_per_step`: Samples per step to be used to approximate expectation. 
- `maxiters`: Number of iteratios for which ADVI should run. ADVI does not currently 
include convergence criteria. As such, the algorithm will run for the full `maxiters` iterations
- `args...`: Other argumetns to be passed on to advi 
- `kwargs...`: Other arguments to be passed on to advi

"""
function advi(bnn::BNN, samples_per_step::Int, maxiters::Int, args...; kwargs...)
    getq(θ) = MvNormal(θ[1:bnn.num_total_params], exp.(θ[bnn.num_total_params+1:end]))
    return advi(bnn, getq, samples_per_step, maxiters, args...; kwargs...)
end

"""
    advi(bnn::BNN, getq::Function, samples_per_step::Int, maxiters::Int; showprogress = true)

Estimate a BNN using Automatic Differentiation Variational Inference (ADVI). 

# Uses
- Uses ADVI as implemented in AdvancedVI.jl

# Arguments 
- `bnn`: A Bayesian Neural Network
- `getq`: A function that takes a vector and returns the variational distribution. 
- `samples_per_step`: Samples per step to be used to approximate expectation. 
- `maxiters`: Number of iteratios for which ADVI should run. ADVI does not currently 
include convergence criteria. As such, the algorithm will run for the full `maxiters` iterations

# Keyword Arguments
- `showprogress = true`: Should progress be shown? 

"""
function advi(bnn::BNN, getq::Function, samples_per_step::Int, maxiters::Int; showprogress = true)
    AdvancedVI.turnprogress(showprogress)
    lπ(θ) = loglikeprior(bnn, θ, bnn.x, bnn.y)
    θnet, θhyper, θlike = bnn.init()
    initθ = vcat(θnet, θhyper, θlike)
    AdvancedVI.setadbackend(:zygote)
    advi = ADVI(samples_per_step, maxiters)

    q = AdvancedVI.vi(lπ, advi, getq, initθ)
    return q
end