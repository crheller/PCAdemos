"""
PCA using nonlinear iterative partial least squares.
Compare result with scikit learn
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_ellipse(x, y):
    inds = np.isfinite(x) & np.isfinite(y)
    x= x[inds]
    y = y[inds]
    data = np.vstack((x, y))
    mu = np.mean(data, 1)
    data = data.T - mu

    D, V = np.linalg.eig(np.divide(np.matmul(data.T, data), data.shape[0] - 1))
    # order = np.argsort(D)[::-1]
    # D = D[order]
    # V = abs(V[:, order])

    t = np.linspace(0, 2 * np.pi, 100)
    e = np.vstack((np.sin(t), np.cos(t)))  # unit circle
    VV = np.multiply(V, np.sqrt(D))  # scale eigenvectors
    e = np.matmul(VV, e).T + mu  # project circle back to orig space
    e = e.T

    return e


N = 200
# generate random data with structured covariance
X = np.random.normal(10, 5, (2, N))
cov_mat = np.array([[1, 0.3], [0.3, 1]])
X = np.matmul(cov_mat, X)

# mean center the variables
X_mean = np.mean(X, axis=-1)
X_center = (X.T - X_mean).T

# ======================== DO PCA =============================
# use NIPALS algorithm (nonlinear iterative partial least squares)
pcs = []
x_alg = X_center.copy()
for i in range(0, X_center.shape[0]):
    tol = 1e-7
    max_iter = 100
    th = x_alg[0, :][np.newaxis, :]
    cost = 1
    iteration = 0
    while (cost > tol) & (iteration < max_iter):
        ph = np.matmul(th, x_alg.T) / np.matmul(th, th.T).squeeze()
        ph = ph / np.linalg.norm(ph)
        th_new = np.matmul(x_alg.T, ph.T) / np.matmul(ph, ph.T).squeeze()
        cost = np.linalg.norm(th_new - th.T)
        th = th_new.T
        iteration+=1

    pcs.append(ph)
    # subtract out pc dim
    x_alg = x_alg - np.matmul(np.matmul(X_center.T, ph.T), ph).T

proj1 = np.matmul(np.matmul(X_center.T, pcs[0].T), pcs[0])
proj2 = np.matmul(np.matmul(X_center.T, pcs[1].T), pcs[1])

# plot PCs
f, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.plot(X_center[0, :], X_center[1, :], '.', color='lightgrey')
e1 = compute_ellipse(X_center[0, :], X_center[1, :])
ax.plot(e1[0, :], e1[1, :], color='k', lw=2)
ax.plot(proj1[:, 0], proj1[:, 1], '-', color='cyan', label='nipals, pc1')
ax.plot(proj2[:, 0], proj2[:, 1], '-', color='firebrick', label='nipals, pc2')

ax.legend()

ax.axis('square')

plt.show()