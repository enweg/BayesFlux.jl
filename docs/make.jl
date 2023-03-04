using Documenter
using BayesFlux

push!(LOAD_PATH, "../src/")
makedocs(
    sitename = "BayesFlux.jl Documentation",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Introduction" => [
            "index.md", 
            "introduction/linear-regression.md", 
            "introduction/feedforward.md", 
            "introduction/recurrent.md"
        ],
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
    modules = [BayesFlux]
)

deploydocs(
    repo = "github.com/enweg/BayesFlux.jl.git",
)
