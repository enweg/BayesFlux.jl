
"""
    abstract type MCMCState end

Every MCMC method must be implemented via a MCMCSTate which keeps track of all 
important information. 

# Mandatory Fields

- `samples::Matrix` of dimension num_total_parameter×num_samples
- `nsampled` the number of thus far sampled sampled

# Mandatory Functions
- `update!(sampler, θ, bnn, ∇θ)` where θ the current parameter vector and ∇θ(θ) is a
  function providing gradients. The function must return θ and num_samples so
  far sampled. 
- `initialise!(sampler, θ, numsamples; continue_sampling)` which initialised the
  sampler. If continue_sampling is true, then the final goal is to obtain
  numsamples samples and thus only the remaining ones still need to be sampled.
- `calculate_epochs(sampler, numbatches, numsamples; continue_sampling)` which
  calculates the number of epochs that must be run through in order to obtain
  `numsamples` samples if `numbatches` batches are used. The number of epochs
  must be returned. If `continue_sampling` is true, then the goal is to obtain
  in total `numsamples` samples and thus we only need the number of epochs that
  still need to be run to obtain this total and NOT the number of epochs to
  sample `numsamples` new samples.
"""
abstract type MCMCState end

calculate_epochs(sampler::MCMCState, nbatches, nsamples; continue_sampling = false) = error("$(typeof(sampler)) did not implement a calculate_epochs method. Please consult the documentation for MCMCState")

update!(sampler::MCMCState, θ::AbstractVector{T}, bnn::BNN, ∇θ) where {T} = error("$(typeof(sampler)) has not implemented an update! method. Please consult the documentation for MCMCState")

function mcmc(bnn::BNN, batchsize::Int, numsamples::Int, sampler::MCMCState; 
    shuffle = true, partial = true, showprogress = true, continue_sampling = false)

    if !partial && !shuffle 
        @warn """shuffle and partial should not be both false unless the data is
        perfectly divided by the batchsize. If this is not the case, some data
        would never be considered"""
    end

    # This allows to stop sampling and continue later
    if continue_sampling 
        θ = sampler.samples[:, end]
    else 
        θnet, θhyper, θlike = bnn.init()
        θ = vcat(θnet, θhyper, θlike)
    end

    batcher = Flux.Data.DataLoader((x = bnn.x, y = bnn.y), 
        batchsize = batchsize, shuffle = shuffle, partial = partial)

    num_batches = length(batcher)
    tosample = continue_sampling ? numsamples - sampler.nsampled : numsamples
    prog = Progress(tosample; desc = "Finding Mode...", 
        enabled = showprogress, showspeed = true)

    ∇θ(θ, x, y) = ∇loglikeprior(bnn, θ, x, y; num_batches = num_batches)

    initialise!(sampler, θ, numsamples; continue_sampling = continue_sampling)
    epochs = calculate_epochs(sampler, num_batches, numsamples; continue_sampling = continue_sampling)
    @info "Running for $epochs epochs."

    nsampled = continue_sampling ? sampler.nsampled : 0
    for e=1:epochs
        for (x, y) in batcher 
            nsampled == numsamples && break
            θ = update!(sampler, θ, bnn, θ -> ∇θ(θ, x, y))
            nsampled = sampler.nsampled
            next!(prog)
        end
    end
    
    return sampler.samples
end
