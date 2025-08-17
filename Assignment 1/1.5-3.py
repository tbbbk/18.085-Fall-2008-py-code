import numpy as np
from scipy.sparse import diags


n = 5
e = np.ones(n)

diagonals = [-e, 2*e, -e]
offsets = [-1, 0, 1]
K = diags(diagonals, offsets, shape=(n, n), format='csr') 


eigenvalues, eigenvectors = np.linalg.eig(K.toarray())
print("Eigenvalues:", np.sort(eigenvalues))

k = np.arange(1, n + 1)

expected_eigenvalues = 2 * (1 - np.cos(np.pi * k / (n + 1)))

print("Expected Eigenvalues:", np.sort(expected_eigenvalues))