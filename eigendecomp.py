import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import helpers.rcParams as _
from settings import figpath

if os.path.isdir(figpath)==False:
    os.system(f"mkdir {figpath}")

# generate random data with structured correlations
X = np.random.normal(0, 1, (2, 50)) * np.array([2, 1])[:, np.newaxis]
cov_mat = np.array([[1, 0.3], [0.3, 1]])
X = np.matmul(cov_mat, X)

f, ax = plt.subplots(1, 1, figsize=(3, 3))



f.tight_layout()

f.savefig(os.path.join(figpath, "pca_data.png"), dpi=300)

# compute and plot covariance matrix
cov = np.cov(X)

f, ax = plt.subplots(1, 2, figsize=(6, 3))

ax[0].scatter(X[0, :], X[1, :], s=25, c="grey")
ax[0].axis("equal")
ax[0].set_xlabel(r"$x_1$")
ax[0].set_ylabel(r"$x_2$")
ax[0].set_title(r"Synthetic data, X")

ax[1].set_title(r"Covariance matrix ($\Sigma$)")
sns.heatmap(cov, cmap='Blues', annot=True, 
                    ax=ax[1], vmin=0, vmax=5, 
                    xticklabels=[r"$X_1$", r"$X_2$"], 
                    yticklabels=[r"$X_1$", r"$X_2$"], 
                    cbar=False)
f.tight_layout()
f.savefig(os.path.join(figpath, "data_and_cov_matrix.png"), dpi=300)


# compute eigenvectors and eigenvalues of covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov) 

# sort according to variance explained
sort_args = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, sort_args]
eigenvalues = eigenvalues[sort_args]

# compute % variance explained
var_explained = eigenvalues / np.sum(eigenvalues)

# do sklearn PCA
pca = PCA()
pca.fit(X.T)

print(f"Eigendecomposition, variance explained ratio: {var_explained}")
print(f"sklearn, variance explained ratio: {pca.explained_variance_ratio_}")

# reconstructions
e_reconstructed = X.T @ eigenvectors[:, [0]] @ eigenvectors[:, [0]].T
sk_reconstructed = X.T @ pca.components_[[0], :].T @ pca.components_[[0], :]
print(f"sum of reconstruction differences: {np.sum(e_reconstructed - sk_reconstructed)}")



# compare our PCs to sklearn PCs
dp = np.abs(np.dot(eigenvectors, pca.components_))
f, ax = plt.subplots(1, 3, figsize=(9, 3))
ax[0].set_title("Pairwise dot product")
sns.heatmap(dp, cmap='Blues', annot=True, 
                    ax=ax[0], vmin=0, vmax=1, 
                    xticklabels=[r"$PC_1$", r"$PC_2$"], 
                    yticklabels=[r"$PC_1$", r"$PC_2$"], 
                    cbar=False)
ax[0].set_xlabel("sklearn")
ax[0].set_ylabel("eigendecomposition")

# plot variance explained per PC
ax[1].plot(var_explained, "-o", label="eigendecomposition")
ax[1].plot(pca.explained_variance_ratio_, "-o", markersize=3, label="sklearn")
ax[1].legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper right", fontsize=8)
ax[1].set_ylabel("Fraction variance explained")
ax[1].set_xlabel("PC")
ax[1].set_xticks([0, 1])
ax[1].set_xlim((-0.5, 1.5))
ax[1].set_xticklabels([1, 2])
ax[1].set_ylim((-0.1, 1.1))
ax[1].axhline(0, linestyle="--", color="k")

# reconstructions
ax[2].scatter(e_reconstructed[:, 0], e_reconstructed[:, 1], s=25, label="eigendecomposition")
ax[2].scatter(sk_reconstructed[:, 0], sk_reconstructed[:, 1], s=10, label="sklearn")
ax[2].axis("equal")
ax[2].legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper right", fontsize=8)
ax[2].set_xlabel(r"$x_1$")
ax[2].set_ylabel(r"$x_2$")
ax[2].set_title(r"$PC_1$ reconstruction")

f.tight_layout()

f.savefig(os.path.join(figpath, "eig_vs_sklearn_summary.png"), dpi=300)