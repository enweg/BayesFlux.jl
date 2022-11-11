"""
Use the Dual Average method to tune the stepsize.

The use of the Dual Average method was proposed in:

Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting
Path Lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research,
15, 31.

"""
mutable struct DualAveragingStepSize{T} <: StepsizeAdapter
    l::T

    mu::T
    target_accept::T
    gamma::T
    t::Int
    kappa::T
    error_sum::T
    log_averaged_step::T

    adapt_steps::Int
end

function DualAveragingStepSize(
    initial_step_size::T; 
    target_accept=T(0.65),
    gamma=T(0.05), 
    t0=10, 
    kappa=T(0.75), 
    adapt_steps=1000
) where {T}

    return DualAveragingStepSize(
        initial_step_size, 
        log(10 * initial_step_size),
        target_accept, 
        gamma, 
        t0, 
        kappa, 
        T(0), 
        T(0), 
        adapt_steps
    )
end

function (sadapter::DualAveragingStepSize{T})(s::MCMCState, mh_probability) where {T}
    if sadapter.t > sadapter.adapt_steps
        s.l = exp(sadapter.log_averaged_step)
        return exp(sadapter.log_averaged_step)
    end

    mh_probability = T(mh_probability)
    sadapter.error_sum += sadapter.target_accept - mh_probability
    log_step = sadapter.mu - sadapter.error_sum / sqrt(sadapter.t * sadapter.gamma)
    eta = sadapter.t^(-sadapter.kappa)
    sadapter.log_averaged_step = eta * log_step + (1 - eta) * sadapter.log_averaged_step
    sadapter.t += 1
    sadapter.l = exp(log_step)

    return exp(log_step)
end