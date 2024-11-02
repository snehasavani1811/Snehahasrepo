'''import numpy as np
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7,8,9],[10,11,12]]])
print('last element from 2nd dim: ', arr[1,0,-3])

import numpy as np
a= np.array([[1,2],[3,4]])
b= np.array([[5,6],[7,8]])
print(a)
print(b)
c= a+b
print(c)
print(a)
print(b)
d= a*b
print(d)
e = np.matmul(a,b)
print(e)
print(a**2)

import numpy as np
a = np.array([[2,1], [-4,3]])
b = np.array([11,3])
x = np.linalg.solve(a,b)
print(x)

Ainv = np.linalg.inv(a)
print(np.matmul(Ainv,b))


