using Flux
using ProgressMeter

"""
Find the mode of a BNN. 

Find the mode of a BNN using `optimiser`. Each `optimiser` must have implemented 
a function `step!(optimiser, θ, ∇θ)` which makes one optimisation step given 
gradient function ∇θ(θ) and current parameter vector θ. The function must return 
θ as the first return value and a flag `has_converged` indicating whether the 
optimisation procedure should be stopped. 
"""
abstract type BNNModeFinder end

step!(mf::BNNModeFinder, θ::AbstractVector, ∇θ::Function) = error("$(typeof(mf)) has not implemented the step! function. Please see the documentation for BNNModeFinder")

"""
    find_mode(bnn::BNN, batchsize::Int, epochs::Int, optimiser::BNNModeFinder)

Find the mode of a BNN.

# Arguments

- `bnn::BNN`: A Bayesian Neural Network formed using `BNN`. 
- `batchsize::Int`: Batchsize used for stochastic gradients. 
- `epochs::Int`: Number of epochs to run for.
- `optimiser::BNNModeFinder`: An optimiser.

# Keyword Arguments

- `shuffle::Bool=true`: Should data be shuffled after each epoch?
- `partial::Bool=true`: Is it allowed to use a batch that is smaller than `batchsize`?
- `showprogress::Bool=true`: Show a progress bar? 

"""
function find_mode(
    bnn::BNN, 
    batchsize::Int, 
    epochs::Int, 
    optimiser::BNNModeFinder;
    shuffle=true, 
    partial=true, 
    showprogress=true
)

    if !partial && !shuffle
        @warn """shuffle and partial should not be both false unless the data is
        perfectly divided by the batchsize. If this is not the case, some data
        would never be considered"""
    end

    θnet, θhyper, θlike = bnn.init()
    θ = vcat(θnet, θhyper, θlike)

    batcher = Flux.Data.DataLoader(
        (x=bnn.x, y=bnn.y),
        batchsize=batchsize, 
        shuffle=shuffle, 
        partial=partial
    )

    num_batches = length(batcher)
    prog = Progress(
        num_batches * epochs; 
        desc="Finding Mode...",
        enabled=showprogress, 
        showspeed=true
    )

    ∇θ(θ, x, y) = ∇loglikeprior(bnn, θ, x, y; num_batches=num_batches)

    for e = 1:epochs
        for (x, y) in batcher
            θ, has_converged = step!(optimiser, θ, θ -> ∇θ(θ, x, y))
            if has_converged
                @info "Converged!"
                return θ
            end
            next!(prog)
        end
    end

    @info "Failed to converge."
    return θ

end