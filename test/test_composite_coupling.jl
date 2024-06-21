using Pkg
Pkg.activate("/home/micki/.julia/environments/splines")
using BAT
using ValueShapes
using Distributions
using PyPlot
using Optimisers
using ArraysOfArrays
using InverseFunctions
using LinearAlgebra

using Revise
using EuclidianNormalizingFlows

# Standard Gaussian 
n_smpls = 10^4
n_dims = 2
bench_smpls = randn(n_dims, n_smpls)

d_GS = BAT.GaussianShell(n = n_dims, r = 7)

samples_GS = bat_sample(d_GS, 
                     BAT.IIDSampling(nsamples=n_smpls)
                     ).result


smpls_flat_GS = ValueShapes.flatview(unshaped.(samples_GS.v))  

mean_x = mean(smpls_flat_GS[1,:])
mean_y = mean(smpls_flat_GS[2,:])
std_x  = std(smpls_flat_GS[1,:])
std_y  = std(smpls_flat_GS[2,:])

smpls_flat_GS = EuclidianNormalizingFlows.ScaleShiftTrafo([1/std_x, 1/ std_y], [-mean_x, -mean_y])(smpls_flat_GS)

fig3, ax3 = plt.subplots(1, figsize=(5,5))
ax3.hist2d(smpls_flat_GS[1,:], smpls_flat_GS[2,:], 100, cmap="inferno")

initial_trafo_GS = CouplingRQSpline(2, 10)

nbatches_GS = 100
nepochs_GS = 50
shuffle_samples = true


optimizer_GS = Optimisers.Adam(1f-3)
smpls_GS = nestedview(smpls_flat_GS);

r_GS = EuclidianNormalizingFlows.optimize_whitening(smpls_GS, 
    initial_trafo_GS, 
    optimizer_GS,
    nbatches = nbatches_GS,
    nepochs = nepochs_GS,
    shuffle_samples = shuffle_samples)

trained_trafo_GS = r_GS.result

smpls_transformed_GS = trained_trafo_GS(smpls_flat_GS)

fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[1].hist2d(smpls_flat_GS[1,:], smpls_flat_GS[2,:], 100, cmap="inferno")
ax[2].hist2d(smpls_transformed_GS[1,:], smpls_transformed_GS[2,:], 100, cmap="inferno")

fig2, ax2 = plt.subplots(1, figsize=(8,4))
ax2.plot(1:length(r_GS.negll_history), r_GS.negll_history)
ax2.set_ylabel("Cost")
ax2.set_xlabel("Iteration")


inv_trafo_GS = inverse(trained_trafo_GS)

bench_transformed_GS = inv_trafo_GS(bench_smpls)

fig, ax = plt.subplots(1,2, figsize=(8,4))

ax[1].hist2d(smpls_flat_GS[1,:], smpls_flat_GS[2,:], 100, cmap="inferno")
ax[1].set_xlim([-2.5, 2.5])
ax[1].set_ylim([-2.5, 2.5])
ax[2].hist2d(bench_transformed_GS[1,:], bench_transformed_GS[2,:], 100, cmap="inferno")
ax[2].set_xlim([-2.5, 2.5])
ax[2].set_ylim([-2.5, 2.5])