##
using DelimitedFiles
using Plots
include("minimalESN.jl")

##
#esn = ESN(inSize=1, outSize=1, resSize=1000, leaking_rate=0.5, sparsity=0.6, spectral_radius=0.9)
esn = ESN() # ESN com valores padrões

##
data = readdlm("MackeyGlass_t17.txt")

##
fit!(esn, data[1:2000])

##
preds = predict!(esn, 1000, 0)
preds = preds'


##
plot(data[2001:3000])
plot!(preds)

##
rmse = sqrt( sum((data[2001:3000] - preds).^2) ./ length(preds))
println(rmse)