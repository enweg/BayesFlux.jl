# Gradient Guided Monte Carlo
# Following papers critices sgld and variants relying on Euler-Maruyama scheme
# and instead proposed to use GGMC with stochastic gradients. 
# Garriga-Alonso, A., & Fortuin, V. (2021). Exact langevin dynamics with stochastic gradients. arXiv preprint arXiv:2102.01691.
using LinearAlgebra

U(θ, llike, lpriorθ, y, x) = -llike(θ, y, x) - lpriorθ(θ)
K(m, M) = 1/2 * m' * inv(M) * m


function ggmc(llike::Function, lpriorθ::Function, batchsize::Int, y::Vector{T}, x::Union{Vector{Matrix{T}}, Matrix{T}}, 
              initθ::Vector{T}, maxiter::Int;
              M = diagm(ones(length(initθ))), l = 0.01, β = 0.5, temp = 1, keep_every = 10, 
              adapruns = 5000, κ = 0.5, goal_accept_rate = 0.65, adaptM = false, diagonal_shrink = 0.9, adapth = true, 
              showprogress = true, debug = false, 
              p = Progress(maxiter * floor(Int, length(y)/batchsize); dt=1, desc="GGMC ...", enabled = showprogress)) where {T <: Real}
    

    # We are exposing a parameterisation in terms of learning rate and momentum parameter
    # but the updates work using h and γ. See paper for details
    h = sqrt(l / length(y))
    γ = -sqrt(length(y)/l)*log(β)
    
    θ = copy(initθ)
    if mod(length(y), batchsize) != 0
        @warn "Batchsize does not properly partition data. Some data will be left out in each cycle."
    end
    num_batches = floor(Int64, length(y)/batchsize)
    @info "Using $num_batches batches."
    θprop = copy(θ)
    tot_iters = maxiter*num_batches

    samples = zeros(eltype(initθ), length(initθ), tot_iters + 1) 
    samples[:, 1] = θ
    lastθi = 1 # column of last added samples

    momentum14 = 0.0*similar(θ)
    momentum12 = 0.0*similar(θ)
    momentum34 = 0.0*similar(θ)
    momentum = 0.0*similar(θ)
    Mhalf = cholesky(M, check = false).L
    Minv = inv(M)
    naccepts = 0

    lMH = -U(θ, llike, lpriorθ, y, x) 
    hastings = zeros(tot_iters+1)
    momenta = zeros(length(θ), tot_iters+1)
    for t=1:maxiter
        yshuffel, xshuffel = sgd_shuffle(y, x) 
        for b=1:num_batches 
            a = exp(-γ*h)
            xbatch = sgd_x_batch(xshuffel, b, batchsize)
            ybatch = sgd_y_batch(yshuffel, b, batchsize)
            g = (length(y)/batchsize) * Zygote.gradient(θ -> U(θ, llike, lpriorθ, ybatch, xbatch), θ)[1]
            if any(isnan.(g))
                println(θ)
                @warn "NaN in gradient. Returning results obtained until now."
                return samples[:, 1:lastθi], hastings[1:lastθi], momenta[:, 1:((t-1)*num_batches + b)]
            end

            mass_window_1 = (100 < lastθi <= adapruns/4)
            mass_window_2 = (adapruns/2 < lastθi <= 3/4*adapruns)
            h_window_1 = (0<lastθi<=500)
            h_window_2 = (adapruns/4 < lastθi <= adapruns/2)
            h_window_3 = (3*adapruns/4 < lastθi <= adapruns)
            in_h_window = h_window_1 || h_window_2 || h_window_3
            in_mass_window = mass_window_1 || mass_window_2
            if adaptM && in_mass_window && mod((t-1)*num_batches+b, keep_every) == 0
                # doing very simple adaptation 
                use_samples = mass_window_1 ? samples[:, 100:Int(adapruns/4)] : samples[:, Int(adapruns/2)+1:Int(3*adapruns/4)]
                # M = (lastθi == Int(adapruns/4) || lastθi == Int(3*adapruns/4)) ? diagm(1 ./ sqrt.(vec(var(use_samples; dims = 2)))) : M
                # M = diagm(1 ./ sqrt.(vec(var(use_samples; dims = 2))))
                # Mhalf = sqrt.(M)
                # Minv = inv(M)
                if mass_window_1
                    Minv = cov(use_samples')
                    Minv = (1-diagonal_shrink)*diagm(diag(Minv)) + diagonal_shrink*Minv
                    M = inv(Minv)
                    Mhalf = cholesky(M, check = false).L
                else
                    # TODO: find a better shrinking scheme
                    # probably could shrink less in second window.
                    Minv = cov(use_samples')
                    Minv = (1-diagonal_shrink)*diagm(diag(Minv)) + diagonal_shrink*Minv
                    M = inv(Minv)
                    Mhalf = cholesky(M, check = false).L
                end

                (lastθi == Int(adapruns/4) || lastθi == Int(3/4*adapruns)) && @info "M = $(M), temp = $temp"
            end


            momentum14 .= sqrt(a)*momentum .+ sqrt((1-a)*temp)*Mhalf*randn(length(θ))
            momentum12 .= momentum14 .- h/2 * g
            θ .= θ .+ h*Minv*momentum12
            θprop = θ

            g = (length(y)/batchsize) * Zygote.gradient(θ -> U(θ, llike, lpriorθ, ybatch, xbatch), θ)[1]
            if any(isnan.(g))
                @warn "NaN in gradient. Returning results until obtained until now."
                return samples[:, 1:lastθi], hastings[1:lastθi], momenta[:, 1:((t-1)*num_batches + b)]
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
                    (lastθi > adapruns) && naccepts += 1 
                else 
                    samples[:, lastθi+1] .= samples[:, lastθi] 
                    θ = samples[:, lastθi]
                end
                lastθi += 1

                if  adapth && in_h_window 
                    # adapting step size/learning rate
                    arate = min(1, hastings[lastθi])
                    H = goal_accept_rate - arate 
                    i = h_window_1 ? lastθi : (h_window_2 ? lastθi - floor(Int, adapruns/4) : lastθi - floor(Int, 3*adapruns/4))
                    eta = i^(-κ)
                    h = exp(log(h) - eta*H)

                    (lastθi == Int(adapruns/2) || lastθi == adapruns) && @info "l=$(length(y)*h^2)"
                end

                lMH = -U(θ, llike, lpriorθ, y, x) 

            end

            if debug
                update!(p, (t-1)*num_batches + b, showvalues = [(:arate, naccepts/lastθi), 
                                                            (:iter, (t-1)*num_batches + b), 
                                                            (:samples, lastθi), 
                                                            (:Madapt, adaptM && in_mass_window), 
                                                            (:hadapt, adapth && in_h_window)])
            else 
                next!(p)
            end
        end

    end

    @info "Acceptance Rate: $(naccepts/(size(samples,2) - adapruns))"
    return samples[:, 1:lastθi], hastings[1:lastθi], momenta

end

function ggmc(bnn::BNN, batchsize::Int, initθ::AbstractVector, maxiter::Int; kwargs...)
    llike(θ, y, x) = loglike(bnn, bnn.loglikelihood, θ, y, x)
    lpriorθ(θ) = lprior(bnn, θ)
    return ggmc(llike, lpriorθ, batchsize, bnn.y, bnn.x, initθ, maxiter; kwargs...)
end

function ggmc(bnn::BNN, batchsize::Int, initθ::Vector{Vector{T}}, maxiter::Int, nchains::Int; kwargs...) where {T<:Real}
    chains = Array{Any}(undef, nchains)
    Threads.nthreads() == 1 && @warn "Only one thread available."
    p = Progress(maxiter * floor(Int, length(bnn.y)/batchsize) * nchains; dt=1, desc="GGMC with $nchains chains ...") 
    Threads.@threads for ch=1:nchains
        chains[ch] = ggmc(bnn, batchsize, initθ[ch], maxiter; p = p, kwargs...)
    end

    samples = cat([ch[1] for ch in chains]...; dims = 3)
    hastings = cat([ch[2] for ch in chains]...; dims = 3)
    momenta = cat([ch[3] for ch in chains]...; dims = 3)

    return samples, hastings, momenta
end