# Imports
using Random
using LinearAlgebra
using SparseArrays
using Arpack: eigs

# ESN struct
mutable struct ESN
    inSize::Integer # Tamanho da entrada
    outSize::Integer # Tamanho da saída
    resSize::Integer # Tamanho do reservatório
    leaking_rate::AbstractFloat # leaking rate
    sparsity::AbstractFloat
    spectral_radius::AbstractFloat
    W
    Win
    Wout
    states
    x
    Yt

    function ESN(; inSize::Integer=1, 
        outSize::Integer=1,
        resSize::Integer=500,
        leaking_rate::AbstractFloat=0.5,
        sparsity::AbstractFloat=0.6,
        spectral_radius::AbstractFloat=0.9)

        Random.seed!(42)
        # pesos da entrada
        Win = rand(Float64, (resSize, 1+inSize)) .- 0.5

        # pesos do reservatório
        W = sprand(Float64,resSize,resSize, 1-sparsity)
        W = W - 0.5*(W .!= 0)

        rhoW = 0

        try
            rhoW = maximum(abs.(eigs(W)[1])) # mais rápido
        catch e
            rhoW = maximum(abs.(eigvals(Matrix(W)))) #funciona sempre
        end

        W = W * spectral_radius / rhoW
        
        esn = new()
        esn.inSize = inSize
        esn.outSize = outSize
        esn.resSize = resSize
        esn.leaking_rate = leaking_rate
        esn.sparsity = sparsity
        esn.spectral_radius = spectral_radius
        esn.Win = Win
        esn.W = W

        return esn
    end        
end

function fit!(esn::ESN, data)
    # matriz de estados
    states = zeros(Float64, (1+esn.inSize+esn.resSize, length(data)-1))

    # matriz "target"
    Yt = (data[(2):(length(data))])'

    # rodando o reservatório e armazenando valores na matriz de estados
    x = zeros(Float64, (esn.resSize, 1))

    for t = 1:length(data)-1
        u = data[t]
        x = (1-esn.leaking_rate)*x + esn.leaking_rate*tanh.( esn.Win*[1;u] + esn.W*x )
        states[:, t] = [1;u;x]
    end

    reg = 1e-8  # regularization

    esn.Wout = Yt*states' * pinv(states*states' + reg*I(1+esn.inSize+esn.resSize))
    
    esn.states = states
    esn.x = x
    esn.Yt = Yt

end

function predict!(esn::ESN, steps, v1)
    Y = zeros(esn.outSize, steps)
    u = esn.Yt[length(esn.Yt)]
    x = esn.x
    for t = 1:steps
        x = (1-esn.leaking_rate)*x + esn.leaking_rate*tanh.( esn.Win*[1;u] + esn.W*x )
        y = esn.Wout*[1;u;x]
        Y[:,t] = y
        # generative mode:
        u = y
    end

    return Y
end