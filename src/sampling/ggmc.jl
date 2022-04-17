# Gradient Guided Monte Carlo
# Following papers critices sgld and variants relying on Euler-Maruyama scheme
# and instead proposed to use GGMC with stochastic gradients. 
# Garriga-Alonso, A., & Fortuin, V. (2021). Exact langevin dynamics with stochastic gradients. arXiv preprint arXiv:2102.01691.
using LinearAlgebra

U(θ, llike, lpriorθ, y, x) = -llike(θ, y, x) - lpriorθ(θ)
K(m, M) = 1/2 * m' * inv(M) * m


function ggmc(llike::Function, lpriorθ::Function, batchsize::Int, y::Vector{T}, x::Union{Vector{Matrix{T}}, Matrix{T}}, 
              initθ::Vector{T}, maxiter::Int;
              M = diagm(ones(length(initθ))), l = 0.01, β = 0.5, temp = 1, keep_every = 10) where {T <: Real}
    

    h = sqrt(l / length(y))
    γ = -sqrt(length(y)/l)*log(β)
    
    θ = copy(initθ)
    if mod(length(y), batchsize) != 0
        @warn "Batchsize does not properly partition data. Some data will be left out in each cycle."
    end
    num_batches = floor(Int64, length(y)/batchsize)
    @info "Using $num_batches batches."
    # θprop = zeros(length(θ), num_batches) 
    θprop = copy(θ)
    tot_iters = maxiter*num_batches


    p = Progress(maxiter * num_batches, 1, "GGMC ...")

    samples = zeros(eltype(initθ), length(initθ), tot_iters + 1) 
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
    hastings = zeros(tot_iters+1)
    momenta = zeros(length(θ), tot_iters+1)
    for t=1:maxiter
        yshuffel, xshuffel = sgd_shuffle(y, x) 
        for b=1:num_batches 
            # print("θ = $θ, lp = $(U(θ, llike, lpriorθ, y, x)), K = $(K(momentum, M))")
            # readline()
            xbatch = sgd_x_batch(xshuffel, b, batchsize)
            ybatch = sgd_y_batch(yshuffel, b, batchsize)
            # printstyled("#### 11111111111\n", color = :light_magenta)
            # println("θ = $θ")
            g = (length(y)/batchsize) * Zygote.gradient(θ -> U(θ, llike, lpriorθ, ybatch, xbatch), θ)[1]
            # println("g = $g")
            # println("Ufull = $(U(θ, llike, lpriorθ, y, x))")
            # println("Ubatch = $(U(θ, llike, lpriorθ, ybatch, xbatch))")
            # println("ybatch = $ybatch")
            # println("xbatch = $xbatch")
            if any(isnan.(g))
                println(θ)
                # error("NaN in gradient. This is likely due to a too large step size. Try using different parameter settings")
                @warn "NaN in gradient. Will turn back to old θ"
                return samples[:, 1:lastθi], hastings[1:lastθi], momenta[:, 1:((t-1)*num_batches + b)]
                if mod((t-1)*num_batches + b, keep_every) == 0
                    samples[:, lastθi+1] = samples[:, lastθi-1]
                    θ = samples[:, lastθi-1]
                    lastθi += 1
                end 
                continue
            end

            adapruns = 2000
            if lastθi > 100 && lastθi <= adapruns/2 && mod((t-1)*num_batches+b, keep_every) == 0
                # doing very simple adaptation 
                # TODO: check literature for better way
                α = 0.0
                Madap = diagm(1 ./ sqrt.(vec(var(samples; dims = 2))))
                M = α*M + (1-α)*Madap
                # println(M)
                # Mhalf = cholesky(M).L
                Mhalf = sqrt.(M)
                lastθi == Int(adapruns/2) && println("final M = $(diag(M))")
            end


            momentum14 .= sqrt(a)*momentum .+ sqrt((1-a)*temp)*Mhalf*randn(length(θ))
            momentum12 .= momentum14 .- h/2 * g
            # println("m14 = $momentum14")
            # println("m12 = $momentum12")
            # println("gradient before update: $g")
            θ .= θ .+ h*inv(M)*momentum12
            θprop = θ

            # printstyled("#### 22222222222222222\n", color = :light_magenta)
            # println("θ = $θ")
            g = (length(y)/batchsize) * Zygote.gradient(θ -> U(θ, llike, lpriorθ, ybatch, xbatch), θ)[1]
            # println("g = $g")
            # println("Ufull = $(U(θ, llike, lpriorθ, y, x))")
            # println("Ubatch = $(U(θ, llike, lpriorθ, ybatch, xbatch))")
            # println("ybatch = $ybatch")
            # println("xbatch = $xbatch")
            if any(isnan.(g))
                println(θ)
                # println(hastings[1:lastθi])
                # error("NaN in gradient 2. This is likely due to a too large step size. Try using different parameter settings")
                @warn "NaN in gradient. Will turn back to old θ"
                return samples[:, 1:lastθi], hastings[1:lastθi], momenta[:, 1:((t-1)*num_batches + b)]
                if mod((t-1)*num_batches + b, keep_every) == 0
                    samples[:, lastθi+1] = samples[:, lastθi-1]
                    θ = samples[:, lastθi-1]
                    lastθi += 1
                end 
                continue
            end
            momentum34 .= momentum12 .- h/2 * g 
            momentum .= sqrt(a)*momentum34 .+ sqrt((1-a)*temp)*Mhalf*randn(length(θ))
            momenta[:, (t-1)*num_batches + b] = momentum

            lMH += K(momentum34, M) - K(momentum14, M)

            if mod((t-1)*num_batches + b, keep_every) == 0
                # time to evaluate MH ratio and keep last draw or start new 
                lMH += U(θ, llike, lpriorθ, y, x)
                lMH /= -temp
                r = rand()
                hastings[lastθi+1] = exp(lMH)
                if r < exp(lMH)
                    samples[:, lastθi+1] .= θprop
                    naccepts += 1 
                else 
                    samples[:, lastθi+1] .= samples[:, lastθi] 
                    θ = samples[:, lastθi]
                end
                lastθi += 1

                if  lastθi <= adapruns
                    # adapting step size/learning rate
                    arate = min(1, hastings[lastθi])
                    H = 0.65 - arate 
                    eta = lastθi^(-0.5)
                    h = exp(log(h) - eta*H)
                    # println(h)

                    lastθi == adapruns && @info "Final l=$(length(y)*h^2)"
                end

                lMH = -U(θ, llike, lpriorθ, y, x) 

            end

            update!(p, (t-1)*num_batches + b, showvalues = [(:arate, naccepts/lastθi), 
                                                            (:iter, (t-1)*num_batches + b), 
                                                            (:samples, lastθi)])
        end

        # lMH += U(θ, llike, lpriorθ, y, x)
        # lMH /= -temp
        # r = rand()
        # hastings[t] = exp(lMH)
        # if r < exp(lMH)
        #     samples[:, lastθi+1:lastθi+num_batches] .= θprop
        #     naccepts += num_batches
        # else 
        #     samples[:, lastθi+1:lastθi+num_batches] .= samples[:, lastθi] 
        #     θ = samples[:, lastθi]
        # end
        # lastθi += num_batches
        # lMH = -U(θ, llike, lpriorθ, y, x) 
    end

    @info "Acceptance Rate: $(naccepts/size(samples,2))"
    return samples[:, 1:lastθi], hastings[1:lastθi], momenta

end

function ggmc(bnn::BNN, batchsize::Int, initθ::AbstractVector, maxiter::Int; kwargs...)
    llike(θ, y, x) = loglike(bnn, bnn.loglikelihood, θ, y, x)
    lpriorθ(θ) = lprior(bnn, θ)
    return ggmc(llike, lpriorθ, batchsize, bnn.y, bnn.x, initθ, maxiter; kwargs...)
end
