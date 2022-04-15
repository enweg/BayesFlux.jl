# implementation of stochastic gradient langevin dynamics

stepsize(a, b, γ, t) = a*(b+t)^(-γ)

function sgld(llike::Function, lpriorθ::Function, batchsize::Int, y::Vector{T}, x::Union{Vector{Matrix{T}}, Matrix{T}}, 
                       initθ::AbstractVector, maxiter::Int;
                       stepsize_a=1.0, stepsize_b=100.0, stepsize_γ=0.3,
                       showprogress = true, opt = Flux.ADAM(), verbose = true) where {T<:Real}
    
    θ = copy(initθ)

    if mod(length(y), batchsize) != 0
        @warn "Batchsize does not properly partition data. Some data will be left out in each cycle."
    end
    num_batches = floor(Int64, length(y)/batchsize)

    p = Progress(maxiter * num_batches, 1, "SGLD ...")

    samples = zeros(eltype(initθ), length(initθ), maxiter*num_batches) 


    for t=1:maxiter
        yshuffel, xshuffel = sgd_shuffle(y, x) 
        for b=1:num_batches 
            xbatch = sgd_x_batch(xshuffel, b, batchsize)
            ybatch = sgd_y_batch(yshuffel, b, batchsize)
            g = (length(y)/batchsize) * Zygote.gradient(θ -> llike(θ, ybatch, xbatch) + lpriorθ(θ), θ)[1]

            stepsize = stepsize_a*(stepsize_b+1)^(-stepsize_γ)
            g = stepsize/2 .* g .+ sqrt(stepsize)*randn(length(θ))
            
            if any(isnan.(g))
                error("NaN in gradient. This is likely due to a too large step size. Try different stepsize parameters (a, b, γ) or a different starting value.")
            end

            θ .+= g

            samples[:, (t-1)*num_batches + b] = copy(θ)
            update!(p, (t-1)*num_batches + b)
        end
    end

    return samples

end 

function sgld(bnn::BNN, batchsize::Int, initθ::AbstractVector, maxiter::Int; kwargs...)
    llike(θ, y, x) = loglike(bnn, bnn.loglikelihood, θ, y, x)
    lpriorθ(θ) = lprior(bnn, θ)
    return sgld(llike, lpriorθ, batchsize, bnn.y, bnn.x, initθ, maxiter; kwargs...)
end