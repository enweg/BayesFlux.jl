# Gradient Guided Monte Carlo
# Following papers critices sgld and variants relying on Euler-Maruyama scheme
# and instead proposed to use GGMC with stochastic gradients. 
# Garriga-Alonso, A., & Fortuin, V. (2021). Exact langevin dynamics with stochastic gradients. arXiv preprint arXiv:2102.01691.
using LinearAlgebra

U(θ, llike, lpriorθ, y, x) = -llike(θ, y, x) - lpriorθ(θ)
K(m, M) = 1/2 * m' * inv(M) * m


function ggmc(llike::Function, lpriorθ::Function, batchsize::Int, y::Vector{T}, x::Union{Vector{Matrix{T}}, Matrix{T}}, 
              initθ::Vector{T}, maxiter::Int;
              M = diagm(ones(length(initθ))), h = 0.01, γ = 1.5, temp = 1) where {T <: Real}
    
    θ = copy(initθ)
    θprop = copy(initθ)
    if mod(length(y), batchsize) != 0
        @warn "Batchsize does not properly partition data. Some data will be left out in each cycle."
    end
    num_batches = floor(Int64, length(y)/batchsize)

    p = Progress(maxiter * num_batches, 1, "GGMC ...")

    samples = zeros(eltype(initθ), length(initθ), maxiter*num_batches) 

    momentum14 = 0.0*similar(θ)
    momentum12 = 0.0*similar(θ)
    momentum34 = 0.0*similar(θ)
    momentum = 0.0*similar(θ)
    Mhalf = cholesky(M).L
    a = exp(-γ*h)
    yshuffel, xshuffel = sgd_shuffle(y, x) 
    naccepts = 0
    for t=1:maxiter
        for b=1:num_batches 
            xbatch = sgd_x_batch(xshuffel, b, batchsize)
            ybatch = sgd_y_batch(yshuffel, b, batchsize)
            g = (length(y)/batchsize) * Zygote.gradient(θ -> llike(θ, ybatch, xbatch) + lpriorθ(θ), θ)[1]
            if any(isnan.(g))
                error("NaN in gradient. This is likely due to a too large step size. Try different stepsize parameters (a, b, γ) or a different starting value.")
            end

            momentum14 = sqrt(a)*momentum + sqrt((1-a)*temp)*Mhalf*randn(length(θ))
            momentum12 = momentum14 - h/2 * g
            θprop = θ + h*inv(M)*momentum12

            g = (length(y)/batchsize) * Zygote.gradient(θ -> llike(θ, ybatch, xbatch) + lpriorθ(θ), θprop)[1]
            if any(isnan.(g))
                error("NaN in gradient. This is likely due to a too large step size. Try different stepsize parameters (a, b, γ) or a different starting value.")
            end
            momentum34 = momentum12 - h/2 * g 
            momentum = sqrt(a)*momentum34 + sqrt((1-a)*temp)*Mhalf*randn(length(θ))
            
            lMH = -1/temp*(U(θprop, llike, lpriorθ, y, x) - U(θ, llike, lpriorθ, y, x) + K(momentum34, M) - K(momentum14, M))
            if (isinf(lMH))
                @warn "Ran into a numerical error. Trying to continue."
            end
            r = rand()
            if r < exp(lMH)
                # accepting
                θ = θprop
                naccepts += 1
            end

            samples[:, (t-1)*num_batches + b] = copy(θ)
            update!(p, (t-1)*num_batches + b)
        end
    end

    @info "Acceptance Rate: $(naccepts/size(samples,2))"
    return(samples)

end

function ggmc(bnn::BNN, batchsize::Int, initθ::AbstractVector, maxiter::Int; kwargs...)
    llike(θ, y, x) = loglike(bnn, bnn.loglikelihood, θ, y, x)
    lpriorθ(θ) = lprior(bnn, θ)
    return ggmc(llike, lpriorθ, batchsize, bnn.y, bnn.x, initθ, maxiter; kwargs...)
end
