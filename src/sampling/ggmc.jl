# Gradient Guided Monte Carlo
# Following papers critices sgld and variants relying on Euler-Maruyama scheme
# and instead proposed to use GGMC with stochastic gradients. 
# Garriga-Alonso, A., & Fortuin, V. (2021). Exact langevin dynamics with stochastic gradients. arXiv preprint arXiv:2102.01691.
using LinearAlgebra

U(θ, llike, lpriorθ, y, x) = -llike(θ, y, x) - lpriorθ(θ)
K(m, M) = 1/2 * m' * inv(M) * m


function ggmc(llike::Function, lpriorθ::Function, batchsize::Int, y::Vector{T}, x::Union{Vector{Matrix{T}}, Matrix{T}}, 
              initθ::Vector{T}, maxiter::Int;
              M = diagm(ones(length(initθ))), h = 0.2, γ = 1.5, temp = 1) where {T <: Real}
    
    θ = copy(initθ)
    if mod(length(y), batchsize) != 0
        @warn "Batchsize does not properly partition data. Some data will be left out in each cycle."
    end
    num_batches = floor(Int64, length(y)/batchsize)
    @info "Using $num_batches batches."
    θprop = zeros(length(θ), num_batches) 

    p = Progress(maxiter * num_batches, 1, "GGMC ...")

    samples = zeros(eltype(initθ), length(initθ), maxiter*num_batches + 1) 
    samples[:, 1] = θ

    momentum14 = 0.0*similar(θ)
    momentum12 = 0.0*similar(θ)
    momentum34 = 0.0*similar(θ)
    momentum = 0.0*similar(θ)
    Mhalf = cholesky(M).L
    a = exp(-γ*h)
    naccepts = 0

    lMH = -U(θ, llike, lpriorθ, y, x) 
    lastθi = 1 # column of last added samples
    hastings = zeros(maxiter)
    for t=1:maxiter
        yshuffel, xshuffel = sgd_shuffle(y, x) 
        for b=1:num_batches 
            xbatch = sgd_x_batch(xshuffel, b, batchsize)
            ybatch = sgd_y_batch(yshuffel, b, batchsize)
            # g = (length(y)/batchsize) * Zygote.gradient(θ -> llike(θ, ybatch, xbatch) + lpriorθ(θ), θ)[1]
            g = (length(y)/batchsize) * Zygote.gradient(θ -> U(θ, llike, lpriorθ, ybatch, xbatch), θ)[1]
            if any(isnan.(g))
                println(θ)
                error("NaN in gradient. This is likely due to a too large step size. Try using different parameter settings")
            end

            if lastθi > 2 & (t-1)*num_batches + b < 1000
                # doing very simple adaptation 
                # TODO: check literature for better way
                α = 0.0
                Madap = diagm(1 ./ vec(var(samples; dims = 2)))
                M = α*M + (1-α)*Madap
                # println(M)
                # Mhalf = cholesky(M).L
                Mhalf = sqrt.(M)
            end


            momentum14 .= sqrt(a)*momentum .+ sqrt((1-a)*temp)*Mhalf*randn(length(θ))
            momentum12 .= momentum14 .- h/2 * g
            # println("gradient before update: $g")
            θ .= θ .+ h*inv(M)*momentum12
            θprop[:,b] = θ

            # g = (length(y)/batchsize) * Zygote.gradient(θ -> llike(θ, ybatch, xbatch) + lpriorθ(θ), θ)[1]
            g = (length(y)/batchsize) * Zygote.gradient(θ -> U(θ, llike, lpriorθ, ybatch, xbatch), θ)[1]
            if any(isnan.(g))
                println(θ)
                error("NaN in gradient. This is likely due to a too large step size. Try using different parameter settings")
            end
            momentum34 .= momentum12 .- h/2 * g 
            momentum .= sqrt(a)*momentum34 .+ sqrt((1-a)*temp)*Mhalf*randn(length(θ))

            lMH += K(momentum34, M) - K(momentum14, M)

            update!(p, (t-1)*num_batches + b, showvalues = [(:arate, naccepts/((t-1)*num_batches + b)), 
                                                            (:iter, (t-1)*num_batches + b)])
        end

        lMH += U(θ, llike, lpriorθ, y, x)
        lMH /= -temp
        r = rand()
        hastings[t] = exp(lMH)
        if r < exp(lMH)
            samples[:, lastθi+1:lastθi+num_batches] .= θprop
            naccepts += num_batches
        else 
            samples[:, lastθi+1:lastθi+num_batches] .= samples[:, lastθi] 
            θ = samples[:, lastθi]
        end
        lastθi += num_batches
        lMH = -U(θ, llike, lpriorθ, y, x) 
    end

    @info "Acceptance Rate: $(naccepts/size(samples,2))"
    return samples, hastings

end

function ggmc(bnn::BNN, batchsize::Int, initθ::AbstractVector, maxiter::Int; kwargs...)
    llike(θ, y, x) = loglike(bnn, bnn.loglikelihood, θ, y, x)
    lpriorθ(θ) = lprior(bnn, θ)
    return ggmc(llike, lpriorθ, batchsize, bnn.y, bnn.x, initθ, maxiter; kwargs...)
end
