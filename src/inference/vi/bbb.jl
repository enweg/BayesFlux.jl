using StatsFuns
using Random, Distributions
using ProgressMeter

function loss_bbb(bnn::B, ψ::AbstractVector{T},
    x::Union{Vector{Matrix{T}},Matrix{T},Array{T,3}},
    y::Union{Vector{T},Matrix{T}}; num_batches=T(1)) where {B<:BNN,T}

    n = Int(length(ψ) / 2)
    μ = ψ[1:n]
    σ = log1pexp.(ψ[n+1:end])
    ϵ = randn(T, n)
    θ = μ .+ σ .* ϵ
    return logpdf(MvNormal(μ, σ), θ) - loglikeprior(bnn, θ, x, y; num_batches=num_batches)
end

function ∇bbb(bnn::B, ψ::AbstractVector{T},
    x::Union{Vector{Matrix{T}},Matrix{T},Array{T,3}},
    y::Union{Vector{T},Matrix{T}}; num_batches=T(1)) where {B<:BNN,T}

    return Zygote.gradient(ψ -> loss_bbb(bnn, ψ, x, y; num_batches=num_batches), ψ)[1]
end

"""
    bbb(args...; kwargs...)

Use Bayes By Backprop to find Variational Approximation to BNN. 

This was proposed in Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra,
D. (2015, June). Weight uncertainty in neural network. In International
conference on machine learning (pp. 1613-1622). PMLR.

# Arguments
- `bnn::BNN`: The Bayesian NN 
- `batchsize::Int`: Batchsize 
- `epochs::Int`: Epochs 

# Keyword Arguments

- `mc_samples::Int=1`: Over how many gradients should be averaged?
- `shuffle::Bool=true`: Should observations be shuffled after each epoch?
- `partial::Bool=true`: Can the last batch be smaller than batchsize? 
- `showprogress::Bool=true`: Show progress bar? 
- `opt=Flux.ADAM()`: Must be an optimiser of type Flux.Optimiser 
- `n_samples_convergence::Int=10`: After each epoch the loss is calculated and
  kept track of using an average of `n_samples_convergence` samples. 
"""
function bbb(
    bnn::BNN, 
    batchsize::Int, 
    epochs::Int;
    mc_samples=1, 
    shuffle=true, 
    partial=true,
    showprogress=true, 
    opt=Flux.ADAM(), 
    n_samples_convergence=10
)

    T = eltype(bnn.like.nc.θ)

    if !partial && !shuffle
        @warn """shuffle and partial should not be both false unless the data is
        perfectly divided by the batchsize. If this is not the case, some data
        would never be considered"""
    end

    θnet, θhyper, θlike = bnn.init()
    initμ = vcat(θnet, θhyper, θlike)
    initσ = ones(T, length(initμ))
    ψ = vcat(initμ, initσ)

    batcher = Flux.Data.DataLoader((x=bnn.x, y=bnn.y),
        batchsize=batchsize, shuffle=shuffle, partial=partial)

    num_batches = length(batcher)
    prog = Progress(num_batches * epochs; enabled=showprogress,
        desc="BBB...", showspeed=true)

    ∇ψ(ψ, x, y) = ∇bbb(bnn, ψ, x, y; num_batches=num_batches)

    ψs = Array{T}(undef, length(ψ), epochs)
    losses = Array{T}(undef, epochs)

    gs = Array{T}(undef, length(ψ), mc_samples)
    for e = 1:epochs
        for (x, y) in batcher
            # Threads.@threads for s=1:mc_samples
            @simd for s = 1:mc_samples
                gs[:, s] = ∇ψ(ψ, x, y)
            end
            g = vec(mean(gs; dims=2))
            Flux.update!(opt, ψ, g)
            next!(prog)
        end
        ψs[:, e] = copy(ψ)
        losses[e] = mean([loss_bbb(bnn, ψ, bnn.x, bnn.y) for _ in 1:n_samples_convergence])
    end

    n = bnn.num_total_params
    μ = ψ[1:n]
    σ = log1pexp.(ψ[(n+1):end])
    return MvNormal(μ, σ), ψs, losses
end
