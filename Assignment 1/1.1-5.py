import numpy as np
from rich import print


K_5_inv = (1 / 6) * np.array([
        [5, 4, 3, 2, 1],
        [4, 8, 6, 4, 2],
        [3, 6, 9, 6, 3],
        [2, 4, 6, 8, 4],
        [1, 2, 3, 4, 5],
    ])

K = np.linalg.inv(K_5_inv)
det_K = np.linalg.det(K)

print("det(K): ", det_K)
print("inv(K): \n", K_5_inv)
print("det(K) * inv(K): \n", det_K * K_5_inv)
