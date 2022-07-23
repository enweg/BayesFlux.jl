using Documenter
using BFlux

push!(LOAD_PATH, "../src/")
makedocs(
    sitename = "BFlux.jl Documentation",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "index.md", 
        "Bayesian Neural Networks" => [
            "model/bnn.md", 
            "model/sampling.md"
        ],
        "Likelihood Functions" => [
            "likelihoods/interface.md", 
            "likelihoods/feedforward.md", 
            "likelihoods/seqtoone.md"
        ],
        "Network Priors" => [
            "priors/interface.md", 
            "priors/gaussian.md", 
            "priors/mixturescale.md"
        ],
        "Inference" => [
            "inference/map.md", 
            "inference/mcmc.md", 
            "inference/vi.md"
        ], 
        "Initialisation" => ["initialise/init.md"], 
        "Utils" => ["utils/recurrent.md"]
    ],
    modules = [BFlux]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
