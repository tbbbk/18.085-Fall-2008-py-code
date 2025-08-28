import numpy as np


V = np.fliplr(np.vander(np.arange(50) / 39))
A = V[:, :12]
b = np.cos(np.arange(0, 3.92+0.08, 0.08))

# Part 1
u_hat = np.linalg.solve(A.T @ A, A.T @ b)

print("Part 1: ", u_hat)

# Part 2
Q, R = np.linalg.qr(A)
u_hat = np.linalg.solve(R, Q.T @ b)

print("Part 2: ", u_hat)
