from scipy.linalg import eigh
import numpy as np
A = np.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]])
B = np.array([0, 1, -1])
C = np.array(B + [1])
D = np.array(B + [1]).mean()
print((B.reshape(-1, 1)==B).astype(np.int))
# print(np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4))))

# print(np.matmul(A, v[:,0]) == w[1] * np.matmul(B, v[:,0]))