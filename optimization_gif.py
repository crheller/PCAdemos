"""
Formulate PCA as a reconstruction optimization problem.
    -- That is, minimize the difference between reconstructed and actual data
    subject to the constraint that the PCs have to be orthonormal

Do this using an iterative fitting procedure where we "subtract out" consecutive PCs
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import helpers.rcParams as _
from helpers.optim import nmse, reconstruct
from settings import figpath

np.random.seed(123)

if os.path.isdir(figpath)==False:
    os.system(f"mkdir {figpath}")

# generate random data with structured correlations
X = np.random.normal(0, 1, (2, 100)) * np.array([2, 1])[:, np.newaxis]
cov_mat = np.array([[1, 0.3], [0.3, 1]])
X = np.matmul(cov_mat, X)

# define a callback function to monitor fitting progress
def callbackF(Xi):
    global Nfeval
    global loss
    global parms
    parms.append(Xi)
    ll = nmse(Xi, Xfit)
    loss.append(ll)
    print(f"Nfeval: {Nfeval}, loss: {ll}")
    Nfeval += 1

# fit model -- iterate over components, fit, deflate, fit next PC
n_components = 2
components_ = np.zeros((n_components, X.shape[0]))
Xfit = X.copy() # we iteratively deflate X while fitting to ensure fitted PCs are orthogonal
# save loss / parameters during fit
loss_optim = []
params_optim = []
for component in range(0, n_components):
    Nfeval = 1
    loss = []
    parms = []

    # initialize the PC
    x0 = np.random.normal(0, 1, X.shape[0])
    x0 = x0 / np.linalg.norm(x0)
    
    # find optimal PC using scpiy's minimize
    result = minimize(nmse, x0, args=(Xfit,), callback=callbackF, method='Nelder-Mead')
    
    # deflate X
    Xfit = Xfit - reconstruct(result.x, Xfit)

    # save resulting PC
    components_[component, :] = result.x
    
    # save all intermediate PCs and loss during the optimization procedure
    loss_optim.append(loss)
    params_optim.append(parms)
    
    

# compare components with sklearn components
pca = PCA()
pca.fit(X.T)
skprojection = reconstruct(pca.components_[0, :], X)

# save video of fitting
component = 0
niter = len(loss_optim[component])
for ii in np.arange(0, niter, step=1):
    f, ax = plt.subplots(1, 2, figsize=(8, 4))
    
    ax[0].scatter(X[0, :], X[1, :], c="grey", s=25)
    pc = params_optim[component][ii]
    pc = pc / np.linalg.norm(pc)
    projection = reconstruct(pc, X)
    ax[0].plot(skprojection[0, :], skprojection[1, :], "k-", label=r"sklearn $PC_1$" ,c="r")
    ax[0].plot(projection[0, :], projection[1, :], "k-", label=r"fit $PC_1$")
    ax[0].axis("equal")
    ax[0].set_xlim((-7, 5))
    ax[0].set_ylim((-6, 6))
    ax[0].set_xlabel(r"$x_1$")
    ax[0].set_ylabel(r"$x_2$")
    ax[0].legend(frameon=False, bbox_to_anchor=(0, 1), loc="upper left")

    ax[1].plot(range(0, ii), loss_optim[component][0:ii])
    ax[1].set_xlim((0, niter))
    ax[1].set_ylim((-0.05, np.max(loss_optim[component])+0.05))
    ax[1].set_ylabel("Reconstruction error")

    f.suptitle(f"N feval: {ii}")
    f.tight_layout()
    nn = str(ii).zfill(3)
    f.savefig(os.path.join(figpath, f"img_{nn}.png"), dpi=300)
    plt.close("all")


np.dot(pca.components_, components_.T)