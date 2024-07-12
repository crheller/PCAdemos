import numpy as np

# define objective function
def frob(pc, X):
    # reconstruct rank-1 view of X
    recon = reconstruct(pc, X)
    # compute error (sq. frob. norm of err in reconstruction)
    err = np.linalg.norm(X-recon, ord="fro") ** 2 # sq. forbenius norm
    # normalize by sq. frob norm of data, X
    err = err / (np.linalg.norm(X)**2)
    return err

# wrapper around frob error function that adds a L1 regularization penalty
def sparse_wrapper(pc, X, lam):
    # compute reconstruction error
    err = frob(pc, X)
    # add L1 regularization
    err = err + (np.sum(np.abs(pc)) * lam) #+  (np.sum(pc**2) * 0.01)
    return err

def constraint(pc):
    return np.linalg.norm(pc) - 1

# Do the PC reconstruction
def reconstruct(pc, X):
    return (X.T @ pc[:, np.newaxis] @ pc[np.newaxis, :]).T