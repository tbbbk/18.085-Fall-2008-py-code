import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


n = 1000

e = np.ones(n)


diagonals = [-e, 2*e, -e]
offsets = [-1, 0, 1]
K = diags(diagonals, offsets, shape=(n, n), format='csr') 


u = spsolve(K, e)

print(u)
plt.plot(u)
plt.title('Solution of Ku = e')
plt.xlabel('Index')
plt.ylabel('u')
plt.grid(True)
plt.show()
