import numpy as np
from scipy.sparse import diags


n = 5
e = np.ones(n)

diagonals = [-e, 2*e, -e]
offsets = [-1, 0, 1]
K = diags(diagonals, offsets, shape=(n, n), format='csr') 

E, Q = np.linalg.eigh(K.toarray())

D = np.array([-1, -1, 1, -1, 1])

D_mat = np.diag(D)

DST = Q @ D_mat

rows = np.arange(1, n+1)  
cols = np.arange(1, n+1) 
JK = np.outer(rows, cols) 

print("DST: ", DST)
print("sin(JK * pi/6) / sqrt(3): ", np.sin(JK * np.pi / 6) / np.sqrt(3))

print("DST ^ T: ", DST.T)
print("DST ^ -1: ", np.linalg.inv(DST))
