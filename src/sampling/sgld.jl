# implementation of stochastic gradient langevin dynamics

"""
    stepsize(a, b, γ, t)

Calculate the stepsize. 

Calculates the next stepsize according to ϵₜ = a(b+t)^(-γ). This is a commonly used 
stepsize schedule that meets the criteria ∑ϵₜ = ∞, ∑ϵₜ² < ∞. 

"""
stepsize(a, b, γ, t) = a*(b+t)^(-γ)

"""
    ∇lp(θ, llike, lpriorθ, y, x, nbatches)

Obtain the gradient of the unnormalised log posterior. 

# Arguments
- `θ`: The parameters
- `llike`: The loglikelihood function of the form `llike(θ, y, x)`
- `lpriorθ` The logprior function of the form `lpriorθ(θ)`
- `y`, `x`, the y and x values. Can be a mini-batch
- `numbatches` Total number of batches. Should be 1 if no mini-batching is used
"""
∇lp(θ, llike, lpriorθ, y, x, nbatches) = nbatches * Zygote.gradient(θ -> llike(θ, y, x) + 1/nbatches * lpriorθ(θ), θ)[1]


function sgld(llike::Function, lpriorθ::Function, batchsize::Int, y::Vector{T}, x::Union{Vector{Matrix{T}}, Matrix{T}}, 
                       initθ::AbstractVector, maxiter::Int;
                       stepsize_a=0.1, stepsize_b=100.0, stepsize_γ=0.8, stepsize_gamma=0.8, stepsize_min = 1e-4,
                       showprogress = true, verbose = true, 
                       p = Progress(maxiter * floor(Int, length(y)/batchsize); dt = 1, desc = "SGLD ...", enabled = showprogress)) where {T<:Real}
    
    # Handling no-unicode environments
    stepsize_γ = stepsize_gamma
    θ = copy(initθ)

    if mod(length(y), batchsize) != 0
        @warn "Batchsize does not properly partition data. Some data will be left out in each cycle."
    end
    num_batches = floor(Int64, length(y)/batchsize)

    samples = zeros(eltype(initθ), length(initθ), maxiter*num_batches) 


    for t=1:maxiter
        yshuffel, xshuffel = sgd_shuffle(y, x) 
        for b=1:num_batches 
            xbatch = sgd_x_batch(xshuffel, b, batchsize)
            ybatch = sgd_y_batch(yshuffel, b, batchsize)
            tt = (t-1)*num_batches + b
            α = stepsize(stepsize_a, stepsize_b, stepsize_γ, tt)
            α = max(α, stepsize_min)
            # g = (length(y)/batchsize) * Zygote.gradient(θ -> llike(θ, ybatch, xbatch) + batchsize/length(y) * lpriorθ(θ), θ)[1]
            g = ∇lp(θ, llike, lpriorθ, ybatch, xbatch, num_batches)
            g = clip_gradient_value!(g)
            g = α/2 .* g .+ sqrt(α)*randn(length(θ))
            
            if any(isnan.(g))
                error("NaN in gradient. This is likely due to a too large step size. Try different stepsize parameters (a, b, γ) or a different starting value.")
            end

            θ .+= g

            samples[:, (t-1)*num_batches + b] = copy(θ)
            next!(p)
        end
    end

    return samples

end 

function sgld(bnn::BNN, batchsize::Int, initθ::AbstractVector, maxiter::Int; kwargs...)
    llike(θ, y, x) = loglike(bnn, bnn.loglikelihood, θ, y, x)
    lpriorθ(θ) = lprior(bnn, θ)
    return sgld(llike, lpriorθ, batchsize, bnn.y, bnn.x, initθ, maxiter; kwargs...)
end

function sgld(bnn::BNN, batchsize::Int, initθ::Vector{Vector{T}}, maxiter::Int, nchains::Int; kwargs...) where {T<:Real}
    chains = Array{Any}(undef, nchains)
    Threads.nthreads() == 1 && @warn "Only one threads available."
    p = Progress(maxiter * floor(Int, length(bnn.y)/batchsize) * nchains; dt = 1, desc = "SGLD using $nchains chains ...")
    Threads.@threads for ch=1:nchains
        chains[ch] = sgld(bnn, batchsize, initθ[ch], maxiter; p = p, kwargs...)
    end
    return chains
end