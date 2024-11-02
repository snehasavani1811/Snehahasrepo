'''
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-10, 10, 400)
y1 = 2 * x + 1
y2 = 2 * x + 2
y3 = 2 * x + 3
plt.figure(figsize=(8, 6))
plt.plot(x, y1, linestyle='-', color='black', label='y = 2x + 1')
plt.plot(x, y2, linestyle='--', color='blue', label='y = 2x + 2')
plt.plot(x, y3, linestyle='-.', color='green', label='y = 2x + 3')
plt.title("Graphs of y = 2x + 1, y = 2x + 2, and y = 2x + 3")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.legend()
plt.grid(True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])
plt.figure(figsize=(8, 6))
plt.scatter(x, y, marker='+', color='blue')
plt.title("Scatter Plot of Points (x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
'''


import numpy as np
A = np.array([
    [1, 2, 3],
    [0, 1, 4],
    [5, 6, 0]
])

try:
    A_inv = np.linalg.inv(A)
except np.linalg.LinAlgError:
    print("Matrix A is singular and cannot be inverted.")
    A_inv = None

if A_inv is not None:
    identity_1 = np.dot(A, A_inv)
    identity_2 = np.dot(A_inv, A)
    print("Matrix A:")
    print(A)
    print("\nInverse of A (A_inv):")
    print(A_inv)
    print("\nProduct of A and A_inv (should be identity):")
    print(identity_1)
    print("\nProduct of A_inv and A (should be identity):")
    print(identity_2)
