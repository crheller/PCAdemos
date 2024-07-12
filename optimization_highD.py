"""
Formulate PCA as a reconstruction optimization problem.
    -- That is, minimize the difference between reconstructed and actual data
    subject to the constraint that the PCs have to be orthonormal

Do this using an iterative fitting procedure where we "subtract out" consecutive PCs

Run the algorithm on a "high-D" dataset and see how robustly we can reconstruct the true PCs as a function of variance explained
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import helpers.rcParams as _
from helpers.optim import frob, reconstruct, constraint
from settings import figpath

np.random.seed(123)

if os.path.isdir(figpath)==False:
    os.system(f"mkdir {figpath}")

ndim = 15
v = np.random.normal(0, 1, (ndim, 5)) # create 5 "eigenvectors"
Q, R = np.linalg.qr(v)                # ensure they are orthogonal
v = Q[:, :5]
v = v / np.linalg.norm(v, axis=0)
v = v * ([5, 4, 3, 2, 1] * np.ones(5)) # scale the eigenvectors
cov_mat = v @ v.T

# covariance matrix is outer product of eigenvectors
cov_mat = v @ v.T

# generate data
nSamples = 500
X = np.random.multivariate_normal(np.zeros(ndim), cov_mat, (nSamples,)).T
X = X + np.random.normal(0, 1, X.shape) # bit of noise so not perfectly just 5 dimensions
X = X - X.mean(axis=1, keepdims=True)

# constraining the norm of each pc to be 1
constraints = ({
    'type': 'eq',
    'fun': constraint
})

# define a callback function to monitor fitting progress
def callbackF(Xi):
    global Nfeval
    global loss
    global parms
    parms.append(Xi)
    ll = frob(Xi, Xfit)
    loss.append(ll)
    print(f"Nfeval: {Nfeval}, loss: {ll}")
    Nfeval += 1

# fit model -- iterate over components, fit, deflate, fit next PC
n_components = ndim
components_ = np.zeros((n_components, X.shape[0]))
Xfit = X.copy() # we iteratively deflate X while fitting to ensure fitted PCs are orthogonal
# save loss / parameters during fit
success = np.zeros(n_components).astype(bool)
for component in range(0, n_components):
    print(f"\n component: {component} \n")
    Nfeval = 1
    loss = []
    parms = []

    # initialize the PC
    x0 = np.random.normal(0, 1, X.shape[0])
    x0 = x0 / np.linalg.norm(x0)
    
    # find optimal PC using scpiy's minimize
    result = minimize(frob, x0, args=(Xfit,), callback=callbackF, method='SLSQP', 
                        tol=1e-7, options={"maxiter": 1e5}, constraints=constraints)
    
    # deflate X
    Xfit = Xfit - reconstruct(result.x, Xfit)

    # save resulting PC
    components_[component, :] = result.x
    
    # save success/not
    success[component] = result.success    
    
# compare components with sklearn components
pca = PCA()
pca.fit(X.T)

# compare true to optimized pcs
f, ax = plt.subplots(1, 2, figsize=(8.25, 4))

# scree plot
ax[0].set_title("Scree plot")
ax[0].plot(pca.explained_variance_ratio_ * 100, "o-", markeredgecolor="k")
ax[0].set_xticks(range(ndim))
ax[0].set_xticklabels(np.arange(1, ndim+1, step=1))
ax[0].set_xlabel("Principal component")
ax[0].set_ylabel("% Variance Explained")

conf_mat = np.abs(pca.components_ @ components_.T)
im = ax[1].imshow(conf_mat, cmap="Blues", vmin=0, vmax=1)
ax[1].set_xlabel(r"sklearn loadings ($W_{sklearn}$)")
ax[1].set_ylabel(r"optimized loadings ($W_{opt}$)")
ax[1].set_title(r"Cosine similarity: $|W_{sklearn}W_{opt}^T|$")
ax[1].set_xticks(np.arange(0, ndim, step=1))
ax[1].set_xticklabels(np.arange(1, ndim+1, step=1))
ax[1].set_yticks(np.arange(0, ndim, step=1))
ax[1].set_yticklabels(np.arange(1, ndim+1, step=1))
f.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

f.tight_layout()

f.savefig(os.path.join(figpath, "highD_optim.png"), dpi=300)