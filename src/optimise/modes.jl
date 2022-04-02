# Find the map estimate (mode of the posteior)
using ProgressBars

function find_mode(bnn::BNN, maxiter::Int, ϵ::Float64=0.01; init_θ = randn(bnn.totparams), 
             showprogress = true, opt = Flux.RMSProp(), verbose = true)

    iterator = showprogress ? ProgressBar(1:maxiter) : 1:maxiter
    θ = init_θ
    lj = lp(bnn, θ)
    for t in iterator
        lj_, g = Zygote.withgradient(θ -> lp(bnn, θ), θ)
        Flux.update!(opt, θ, -g[1])
        if 1 < (lj_ / lj) < (1+ϵ)
            if (verbose) @info "Found mode in $t iterations." end
            return θ, true
        end
        lj = lj_
    end
    if (verbose) @info "Did not converge within $maxiter iterations." end
    return θ, false
end

