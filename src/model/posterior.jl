# Methods used for posterior analysis and predictions

function posterior_predict(bnn::BNN, samples::Matrix{T}; kwargs ...) where {T<:Real} 
    return predict(bnn, bnn.loglikelihood, samples; kwargs...)
end