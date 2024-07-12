"""
Basic visualization of PCA
2-D data, project onto basis, reconstruct from basis
"""
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import helpers.rcParams as _
from settings import figpath

# generate data with correlations
X = np.random.normal(0, 1, (2, 50)) * np.array([5, 1])[:, np.newaxis]
cov_mat = np.array([[1, 0.1], [0.5, 1]])
X = np.matmul(cov_mat, X)

# perform PCA
pca = PCA(n_components=2)
pca.fit(X.T)

pc1_projection = X.T.dot(pca.components_[0, :])
pc1_reconstruction = pc1_projection[:, np.newaxis].dot(pca.components_[[0], :])

pc2_projection = X.T.dot(pca.components_[1, :])
pc2_reconstruction = pc2_projection[:, np.newaxis].dot(pca.components_[[1], :])

# plot original data with new basis vectors, transformation, reconstruction
f, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].scatter(X[0, :], X[1, :], s=25, c="grey")
ax[0].plot(pc1_reconstruction[:, 0], pc1_reconstruction[:, 1], "r-", label=r"Basis vector 1 ($PC_1$)")
ax[0].plot(pc2_reconstruction[:, 0], pc2_reconstruction[:, 1], "k-", label=r"Basis vector 2 ($PC_2$)")
ax[0].axis("equal")
ax[0].set_xlabel(r"$x_1$")
ax[0].set_ylabel(r"$x_2$")
ax[0].legend(frameon=False, bbox_to_anchor=(0, 1), loc="lower left")

ax[1].plot(pc1_projection, ".", color="r", label=r"Projection onto $PC_1$")
ax[1].plot(pc2_projection, ".", color="k", label=r"Projection onto $PC_2$")
ax[1].set_xlabel("Observations")
ax[1].set_ylabel(r"Projection onto $PC$")
ax[1].legend(frameon=False, bbox_to_anchor=(0, 1), loc="lower left")

ax[2].scatter(pc1_reconstruction[:, 0], pc1_reconstruction[:, 1], s=25, color="r", label=r"$PC_1$ reconstruction")
ax[2].axis("equal")
ax[2].set_xlabel(r"$x_1$")
ax[2].set_ylabel(r"$x_2$")
ax[2].legend(frameon=False, bbox_to_anchor=(0, 1), loc="lower left")

f.tight_layout()

f.savefig(os.path.join(figpath, "PCA_basic.svg"), dpi=300)