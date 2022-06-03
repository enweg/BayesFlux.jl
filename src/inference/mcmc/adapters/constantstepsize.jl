
"""
Use a contant stepsize.
"""
struct ConstantStepsize{T} <: StepsizeAdapter
    l::T
end

(sadapter::ConstantStepsize{T})(s::MCMCState, mh_probability) = sadapter.l