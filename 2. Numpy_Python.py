#############################################################
# AN INTRODUCTION TO PYTHON NUMPY: ARRAYS AND MATRICES
#############################################################

# Sources:
# Kevin Markham: https://github.com/justmarkham

# NumPy is an extension to the Python programming language, 
# adding support for large, multi-dimensional (numerical)
# arrays and matrices, along with a large library of 
# high-level mathematical functions to operate on these arrays.

#############################################################
#############################################################
# NUMPY
#############################################################
#############################################################

#############################################################
# 1 - START
#############################################################
from __future__ import print_function
import numpy as np

#############################################################
# 2 - CREATE ARRAYS
#############################################################
# Create "ndarrays" from lists. Note: every element must be 
# the same type (will be converted if possible)

# Create a 1d-array
# IMPORTANT: a 1d-array cannot be transposed as it lacks the 
# second dimension. So, it is neither a row-vector, nor a col-vector.
# It has no orientation.        
v1 = np.array([1,2]) # the argument is a list [1,2]
v1.ndim # dim=1

# However, we can add the second dimension by incapsulation using "[]"
v2 = np.array([[1,2]]) # the argument is a list [1,2]
v2.ndim # dim=2
v2.shape # by default, it is now a "row-vector"
# We can now transpose it
v2.T # it is now a "col-vector"

# In general, we can define a 2d-array in this way:
x2 = np.array([[1,2,3],[4,5,6]])
x2.shape 
# (2,3)

# IMPORTANT: if we multiply (see later) a 1d-array 
# (with no orientation) by a 2d-array (that always has orientation),
# Python makes the multiplication by adjusting on the right dimensions.
# For example:
M = np.array([[1,2,3],[4,5,6],[7,8,9]]) # 2d-array
x = np.array([2,4,3]) # 1d-array
# Both multiplications, (1) H=xM and (2) H=Mx, work well:
H = np.dot(M,x)
H
# array([19, 46, 73])
H = np.dot(x,M)
# array([39, 48, 57])

# Suppose, however to give x an orientation, so that it is a 2d-array:
x = x = np.array([[2,4,3]]) # 2d-array
x.shape # by-default x is a (1 x 3) array (i.e., a row-vector)
# This works:
H = np.dot(x,M)
# But this does not work anymore:
H = np.dot(M,x) # returns an error
# But if we transpose x, then it works:
H = np.dot(M,x.T) 
# Also, H is 2d, as expected:
H.shape
# (3, 1)

# We can also genarate a 3d-array by incapsulation using "[]"
v3 = np.array( [  [[1,2],[3,4]] , [[5,6],[7,8]]  ])
v3.ndim # dim=3
v3.shape # shape=(2, 2, 2)

# Other examples
data1 = [1, 2, 3, 4, 5] # list
arr1 = np.array(data1) # 1d array
data2 = [range(1, 5), range(5, 9)] # list of lists
print(data2)
arr2 = np.array(data2) # 2d array
print(arr2)
A=arr2.tolist() # convert array back to list
print(A)

# Create special arrays
np.zeros(10)
np.zeros((3, 6))
np.ones(10)
np.linspace(0, 1, 5) # 0 to 1 (inclusive) with 5 points
np.logspace(0, 3, 4) # 10^0 to 10^3 (inclusive) with 4 points

# "arange" is like "range", except it returns an array (not a list)
int_array = np.arange(5)
float_array = int_array.astype(float)

#############################################################
# 3 - EXAMINING ARRAYS
#############################################################
arr1.dtype # float64
arr2.dtype # int32
arr2.ndim  # 2
arr2.shape # (2, 4) - axis 0 is rows, axis 1 is columns
arr2.size  # 8 - total number of elements
len(arr2)  # 2 - size of first dimension (aka axis)

#############################################################
# 4 - RESHAPING
#############################################################
arr = np.arange(10, dtype=float).reshape((2, 5))
print(arr.shape)
print(arr.reshape(5, 2))

# Add an axis
a = np.array([0, 1])
a_col = a[:, np.newaxis]
a_row = a[np.newaxis,:]
print(a_col)
print(a_row)
#or
a_col = a[:, None]

# Transpose
print(a_col.T)

# Flatten: always returns a flat copy of the orriginal array
arr_flt = arr.flatten()
arr_flt[0] = 33
print(arr_flt)
print(arr)

# Ravel: returns a view of the original array whenever possible.
arr_flt = arr.ravel()
arr_flt[0] = 33
print(arr_flt)
print(arr)

#############################################################
# 5 - STACK ARRAYS
#############################################################
# Stack flat arrays in columns
a = np.array([0, 1])
b = np.array([2, 3])
ab = np.stack((a, b)).T
print(ab)
# or
np.hstack((a[:, None], b[:, None]))

#############################################################
# 6 - SELECTION
#############################################################
# Single item
arr = np.arange(10, dtype=float).reshape((2, 5))
arr[0] # 0th element (slices like a list)
arr[0, 3] # row 0, column 3: ret

#############################################################
# 7 - SLICING
#############################################################
# Syntax: start:stop:step with start (default 0) stop (default last) step (default 1)
arr[0, :]   # row 0: returns 1d array ([0, 1, 2, 3, 4])
arr[:, 0]   # column 0: returns 1d array ([1, 5])
arr[:, :2]  # columns strictly before index 2 (2 first columns)
arr[:, 2:]  # columns after index 2 included
arr2 = arr[:, 1:4] # columns between index 1 (included) and 4 (excluded)
print(arr2)

# Slicing returns a view (not a copy)
arr2[0, 0] = 33
print(arr2)
print(arr)

# Row 0: reverse order
print(arr[0, ::-1])

# The rule of thumb here can be: in the context of lvalue 
# indexing (i.e. the indices are placed in the left hand side 
# value of an assignment), no view or copy of the array is 
# created (because there is no need to). 
# However, with regular values, the above rules for 
# creating views does apply.

#############################################################
# 8. INTEGER OR BOOLEAN ARRAY INDEXING
#############################################################
# Fancy indexing returns a copy not a view

#Integer array indexing
arr2 = arr[:, [1,2,3]] # return a copy
print(arr2)
arr2[0, 0] = 44
print(arr2)
print(arr)

# Boolean arrays indexing
arr2 = arr[arr > 5] # return a copy
print(arr2)
arr2[0] = 44
print(arr2)
print(arr)

# However, In the context of lvalue indexing 
# (left hand side value of an assignment) 
# Fancy authorizes the modification of the original array
arr[arr > 5] = 0
print(arr)


# Boolean arrays indexing continues
names = np.array(['Bob', 'Joe', 'Will', 'Bob'])
names == 'Bob' # returns a boolean array
names[names != 'Bob'] # logical selection
(names == 'Bob') | (names == 'Will') # keywords "and/or" don't work with boolean arrays
names[names != 'Bob'] = 'Joe' # assign based on a logical selection
np.unique(names) # set function


#############################################################
# VECTORIZED OPERATIONS
#############################################################
nums = np.arange(5)
nums * 10 # multiply each element by 10
nums = np.sqrt(nums) # square root of each element
np.ceil(nums) # also floor, rint (round to nearest int)
np.isnan(nums) # checks for NaN
nums + np.arange(5) # add element-wise
np.maximum(nums, np.array([1, -2, 3, -4, 5])) # compare element-wise

# Compute Euclidean distance between 2 vectors
vec1 = np.random.randn(10)
vec2 = np.random.randn(10)
dist = np.sqrt(np.sum((vec1 - vec2) ** 2))

# math and stats
rnd = np.random.randn(4, 2) # random normals in 4x2 array
rnd.mean()
rnd.std()
rnd.argmin() # index of minimum element
rnd.sum()
rnd.sum(axis=0) # sum of columns
rnd.sum(axis=1) # sum of rows

# methods for boolean arrays
(rnd > 0).sum() # counts number of positive values
(rnd > 0).any() # checks if any value is True
(rnd > 0).all() # checks if all values are True

# random numbers
np.random.seed(12234) # Set the seed
np.random.rand(2, 3) # 2 x 3 matrix in [0, 1]
np.random.randn(10) # random normals (mean 0, sd 1)
np.random.randint(0, 2, 10) # 10 randomly picked 0 or 1

#############################################################
# BROADCASTING
#############################################################
# Sources: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html 
# Implicit conversion to allow operations
# on arrays of different sizes. - The smaller array is stretched 
# or “broadcasted” across the larger array so that they
# have compatible shapes. - Fast vectorized operation in C i
# nstead of Python. - No needless copies.

# Rules
# Starting with the trailing axis and working backward, Numpy compares arrays dimensions.
# • If two dimensions are equal then continues
# • If one of the operand has dimension 1 stretches it to match the largest one
# • When one of the shapes runs out of dimensions (because it has less dimensions than the other shape), Numpy
# will use 1 in the comparison process until the other shape’s dimensions run out as well.

a = np.array([[ 0, 0, 0],
[10, 10, 10],
[20, 20, 20],
[30, 30, 30]])
b = np.array([0, 1, 2])
print(a + b)

# Examples
# Shapes of operands A, B and result:
# A (2d array): 5 x 4
# B (1d array): 1
# Result (2d array): 5 x 4
# A (2d array): 5 x 4
# B (1d array): 4
# Result (2d array): 5 x 4
# A (3d array): 15 x 3 x 5
# B (3d array): 15 x 1 x 5
# Result (3d array): 15 x 3 x 5
# A (3d array): 15 x 3 x 5
# B (2d array): 3 x 5
# Result (3d array): 15 x 3 x 5
# A (3d array): 15 x 3 x 5
# B (2d array): 3 x 1
# Result (3d array): 15 x 3 x 5

#############################################################
# END
#############################################################





