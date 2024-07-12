"""
Comparse sparse PCA to normal PCA
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, SparsePCA
from scipy.optimize import minimize
import helpers.rcParams as _
from helpers.optim import frob, reconstruct, sparse_wrapper, constraint

from settings import figpath

np.random.seed(123)

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


# perform PCA to check the ground truth for the data we're working with
# compare components with sklearn fit to ensure we reached the correct solution
pca = PCA()
pca.fit(X.T)

f, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].set_title("Data matrix")
ax[0].imshow(X, aspect="auto", interpolation="none")
ax[1].plot(pca.components_[0, :], label="full, PC1")
ax[1].legend(frameon=False, bbox_to_anchor=(0, 1), loc="lower left")
ax[2].plot(pca.explained_variance_ratio_, "o-", label="full")
ax[2].set_title("Explained variance")
f.tight_layout()


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

# ====================== fit sparse model =======================
lams = [0.01, 0.1, 1] # sparsity penalty
n_components = ndim
sparse_components_ = np.zeros((len(lams), n_components, X.shape[0]))
# save loss / parameters during fit
sparse_loss = []
success = np.zeros((len(lams), n_components))
for i, lam in enumerate(lams):
    print(f"\n lambda: {lam} \n")
    _sparse_loss = []
    Xfit = X.copy()
    for component in range(0, n_components):
        Nfeval = 1
        loss = []
        parms = []

        # initialize the PC
        x0 = pca.components_[component, :]
        x0 = x0 / np.linalg.norm(x0)
        
        # find optimal PC using scpiy's minimize
        result = minimize(sparse_wrapper, x0, args=(Xfit,lam), callback=callbackF, method='SLSQP', 
                        tol=1e-6, options={"maxiter": 1e5}, constraints=constraints)
        sparse_components_[i, component, :] = result.x
        success[i, component] = result.success

        # deflate X
        Xfit = Xfit - reconstruct(result.x, Xfit)
        
        # save all intermediate PCs and loss during the optimzation procedure
        _sparse_loss.append(loss)

    sparse_loss.append(_sparse_loss)


# compute variance explained for each model
tvar = np.sum((X - X.mean(axis=1, keepdims=True))**2, axis=1).sum()
explained_variance_ratio_ = np.zeros((len(lams), ndim))
for component in range(ndim):
    for lam in range(len(lams)):
        comp = sparse_components_[lam, component, :]
        rr = reconstruct(comp, X)
        rvar = np.sum((rr - rr.mean(axis=1, keepdims=True))**2, axis=1).sum()
        fract = rvar / tvar
        explained_variance_ratio_[lam, component] = fract

# heatmap of loadings
loadings1 = np.concatenate((pca.components_[[0], :], sparse_components_[:, 0, :]), axis=0)
loadings2 = np.concatenate((pca.components_[[1], :], sparse_components_[:, 1, :]), axis=0)

f = plt.figure(figsize=(8, 4))
ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
ax2 = plt.subplot2grid((2, 4), (0, 2), colspan=2)
ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)

ax1.plot(pca.explained_variance_ratio_ * 100, "o-", markeredgecolor="k", label=fr"$\lambda$=0")
for i, lam in enumerate(lams):
    ax1.plot(explained_variance_ratio_[i, :] * 100, ".-", markeredgecolor="none", label=fr"$\lambda$={lam}")
ax1.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper right")
ax1.set_xticks(range(ndim))
ax1.set_xticklabels(np.arange(1, ndim+1))
ax1.set_xlabel("Principal component")
ax1.set_ylabel("% variance explained")
ax1.set_title("Scree plot")

ax2.imshow(loadings1, cmap="bwr", vmin=-0.75, vmax=0.75, aspect="auto")
ax2.set_yticks(range(len(lams)+1))
ax2.set_yticklabels(np.append([0], lams))
ax2.set_ylabel(r"$\lambda$")
ax2.set_title(r"$PC_1$ loadings")
ax2.set_xticks([])

ax3.imshow(loadings2, cmap="bwr", vmin=-0.75, vmax=0.75, aspect="auto")
ax3.set_yticks(range(len(lams)+1))
ax3.set_yticklabels(np.append([0], lams))
ax3.set_ylabel(r"$\lambda$")
ax3.set_title(r"$PC_2$ loadings")
ax3.set_xticks([])

f.tight_layout()

f.savefig(os.path.join(figpath, "sparse.png"), dpi=300)
