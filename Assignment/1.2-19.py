import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


h = 1 / 5
n = 4

offsets = [-1, 0, 1]

diagonals_1 = [-1, 2, -1]
K_1 = (1 / (h ** 2)) * diags(diagonals_1, offsets, shape=(n, n), format='csr')

diagonals_2 = [-1, 0, 1]
K_2 = (1 / (2 * h)) * diags(diagonals_2, offsets, shape=(n, n), format='csr')

K_c = K_1 + K_2

e = np.ones(n)

u_centered = spsolve(K_c, e) # Discrete centered finite difference solution

diagonals_3 = [0, -1, 1]
K_3 = (1 / h) * diags(diagonals_3, offsets, shape=(n, n), format='csr')

K_f = K_1 + K_3

u_forward = spsolve(K_f, e) # Discrete forward finite difference solution

print(u_centered)
print(u_forward)

ux = lambda x : x - 1 / (1 - math.e) * (1 - math.e ** x)

x = np.linspace(0, 1, 100)  
x_points = np.array([0.2, 0.4, 0.6, 0.8])  

u_exact = np.array([ux(xi) for xi in x])

plt.figure(figsize=(8, 6))
plt.plot(x_points, u_centered, marker='o', label='Centered Difference')
plt.plot(x_points, u_forward, marker='s', label='Forward Difference')
plt.plot(x, u_exact, label='Exact Solution', linestyle='--')  
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Finite Difference Solutions vs Exact Solution')
plt.legend()
plt.grid(True)
plt.show()