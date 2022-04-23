# Methods used for posterior analysis and predictions

function posterior_predict(bnn::BNN, samples::Matrix{T}; kwargs ...) where {T<:Real} 
    return predict(bnn, bnn.loglikelihood, samples; kwargs...)
end

function posterior_predict(bnn::BNN, samples::Array{T, 3}; kwargs...) where {T<:Real}
    ppreds = [posterior_predict(bnn, samples[:,:,i]; kwargs...) for i=1:size(samples, 3)]
    ppreds = cat(ppreds...; dims = 3)
    return ppreds
end