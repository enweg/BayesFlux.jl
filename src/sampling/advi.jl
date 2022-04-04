using AdvancedVI, DistributionsAD
using ProgressMeter

# ADVI for BFlux

function advi(bnn::BNN, samples_per_step::Int, maxiters::Int, args...; kwargs...)
    getq(θ) = MvNormal(θ[1:bnn.totparams], exp.(θ[bnn.totparams+1:end]))
    return advi(bnn, getq, samples_per_step, maxiters, args...; kwargs...)
end

function advi(bnn::BNN, getq::Function, samples_per_step::Int, maxiters::Int; showprogress = true)
    AdvancedVI.turnprogress(showprogress)
    lπ(θ) = lp(bnn, θ)
    initθ = rand(MvNormal(zeros(bnn.totparams*2), log(10)/3*ones(bnn.totparams*2)))
    println("initθ = $initθ")
    AdvancedVI.setadbackend(:zygote)
    advi = ADVI(samples_per_step, maxiters)

    q = AdvancedVI.vi(lπ, advi, getq, initθ)
    return q
end

# FIXME: This somehow breaks; ADVI does not seem to work in threads. 
function advi(bnn::BNN, getq::Function, samples_per_step::Int, maxiters::Int, num_qs::Int)
    qs = Array{Distribution}(undef, num_qs)
    AdvancedVI.turnprogress(false) # show no individual progress

    p = Progress(num_qs)
    update!(p, 0)
    # jj = Threads.Atomic{Int}(0)
    # l = Threads.SpinLock()

    Threads.@threads for i=1:num_qs
    # for i=1:num_qs
        qs[i] = advi(bnn, getq, samples_per_step, maxiters; showprogress = false)
        # Threads.atomic_add!(jj, 1)
        # Threads.lock(l)
        # update!(p, jj[])
        # Threads.unlock(l)
    end
    return qs
end