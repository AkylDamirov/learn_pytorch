import time
import torch
# print(torch.__version__)

scalar = torch.tensor(7)
# print(scalar)
# print(scalar.ndim)

# Get the Python number within a tensor (only works with one-element tensors)
# print(scalar.item())

#vector
vector = torch.tensor([7,7])
# print(vector)

#check number of dimensions of vector
# print(vector.dim())

#check shape of vector
# print(vector.shape)

#
matrix = torch.tensor([[7,8],
                      [9, 10]])
# print(matrix)
#check numbers of dimensions
# print(matrix.ndim)

#MATRIX has two dimensions (did you count the number of square brakcets on the outside of one side?)
# print(matrix.shape)

#Tensor
Tensor = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])

# print(Tensor)
#check number of dimensions or Tensor
# print(Tensor.ndim)
#
# #and shape
# print(Tensor.shape)
#Alright, it outputs torch.Size([1, 3, 3]). The dimensions go outer to inner. That means there's 1 dimension of 3 by 3.

#Let's summarise.
# scalar	a single number
# vector	a number with direction (e.g. wind speed with direction) but can also have many other numbers
# matrix	a 2-dimensional array of numbers
# tensor	an n-dimensional array of numbers

#How to create tensor with random numbers
#We can do so using torch.rand() and passing in the size parameter.
# random_tensor = torch.rand(size=(3,4))
# print(random_tensor, random_tensor.dtype)

#The flexibility of torch.rand() is that we can adjust the size to be whatever we want.
#For example, say you wanted a random tensor in the common image shape of [224, 224, 3] ([height, width, color_channels]).

random_image_size_tensor = torch.rand(size=(224,224,3))
# print(random_image_size_tensor.shape, random_image_size_tensor.ndim)
# print(random_image_size_tensor)
# print(torch.rand(3,4))


#Zeros and Ones

#lets create a tensor full of zeros with torch.zeros()
# zeros = torch.zeros(size=(3,4))
# print(zeros, zeros.dtype)

#the same with ones
# ones = torch.ones(size=(3,4))
# print(ones, ones.dtype)

#Create a range and tensors like
#use torch.arange(start, end, step)


# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded
#
# print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)

float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16) # torch.half would also work

# print(float_16_tensor.dtype)

#Basic operations (+, -, *)
tensor_m = torch.tensor([1,2,3])
# tensor_m += 10 add for each element
# print(tensor_m)
# tensor_m = tensor_m * 10 #multiply for each element

#pytorch has func for adding torch.multiply(tensor, amount to multiply) and torch.add()

#MATRIX multiplication
#PyTorch implements matrix multiplication functionality in the torch.matmul() method.

# The inner dimensions must match:
# (3, 2) @ (3, 2) won't work
# (2, 3) @ (3, 2) will work
# (3, 2) @ (2, 3) will work
# The resulting matrix has the shape of the outer dimensions:
# (2, 3) @ (3, 2) -> (2, 2)
# (3, 2) @ (2, 3) -> (3, 3)

#Note: "@" in Python is the symbol for matrix multiplication.

#Element-wise multiplication	[1*1, 2*2, 3*3] = [1, 4, 9]	tensor * tensor
#Matrix multiplication	[1*1 + 2*2 + 3*3] = [14]	tensor.matmul(tensor)

tensor = torch.tensor([1, 2, 3])
#element-wise matrix multiplication
# tensor*tensor

# #matrix multiplication
# torch.matmul(tensor, tensor)
#can also use @ for matrix multiplocation
# tensor @ tensor

#you can do multiplication by hand, but its not recommended
#The in-built torch.matmul() method is faster


#matrix multiplication by hand
# avoid doing operations with for loops at all cost, they are computationally expensive
# value = 0
# for i in range(len(tensor)):
#     value += tensor[i]*tensor[i]
# print(value)
#
# a = torch.matmul(tensor, tensor)
# print(a)


#one of the most common errors (shape errors)
#Because much of deep learning is multiplying and performing operations on matrices and matrices have a strict rule about what shapes and sizes can be combined,
# one of the most common errors you'll run into in deep learning is shape mismatches.

tensor_A = torch.tensor([[1,2],
                         [3,4],
                         [5,6]], dtype=torch.float32)

tensor_B = torch.tensor([[7,8],
                         [9,10],
                         [11, 12]], dtype=torch.float32)

# print(torch.matmul(tensor_A, tensor_B)) #this will error

#We can make matrix multiplication work between tensor_A and tensor_B by making their inner dimensions match.
#One of the ways to do this is with a transpose (switch the dimensions of a given tensor).

# You can perform transposes in PyTorch using either:
#
# torch.transpose(input, dim0, dim1) - where input is the desired tensor to transpose and dim0 and dim1 are the dimensions to be swapped.
# tensor.T - where tensor is the desired tensor to transpose

#View tensor_A and tensor_B.T
# print(tensor_A)
# print(tensor_B.T)

#the operation works when tensor_B is transposed
# print(f'original shapes - tensor_A-{tensor_A.shape}, tensor_B-{tensor_B.shape}')
# print(f'New shapes - tensor_A-{tensor_A.shape}, tensor_B.T={tensor_B.T.shape}')
# print(f'Multiply {tensor_A} * {tensor_B.T}')
# print('output')
# output = torch.matmul(tensor_A, tensor_B.T)
# print(output)
# print(f'output shape {output.shape}')

#you can also use torch.mm()

#finding min max mean sum etc
x = torch.arange(0, 100, 10)

# print(f'Minimum {x.min()}')
# print(f'Maximum {x.max()}')
# # print(f'Mean {x.mean()}') #this will error
# print(f'Mean {x.type(torch.float32).mean()}')#wont work without float datatype
# print(f'Sum {x.sum()}')

#You can also do the same as above with torch methods.
#torch.max(x), torch.min(x), torch.mean(x.type(torch.float32)), torch.sum(x)

#how to find inde position min/max
# print(f' index Where min occurs {x.argmin()}')
# print(f' index Where max occurs {x.argmax()}')

#change tensor datatype
# As mentioned, a common issue with deep learning operations is having your tensors in different datatypes.
# If one tensor is in torch.float64 and another is in torch.float32, you might run into some errors.
# But there's a fix.
# You can change the datatypes of tensors using torch.Tensor.type(dtype=None) where the dtype parameter is the datatype you'd like to use.

#create tensor and check its datatype
tensor_c = torch.arange(0., 100., 10.)
# print(tensor_c.dtype)

#create a float16 tensor
# tensor_float16 = tensor_c.type(torch.float16)
# print(tensor_float16)
#
# Note: Different datatypes can be confusing to begin with. But think of it like this, the lower the number (e.g. 32, 16, 8),
# the less precise a computer stores the value.
# And with a lower amount of storage, this generally results in faster computation and a smaller overall model. Mobile-based neural networks
# often operate with 8-bit integers, smaller and faster to run but less accurate than their float32 counterparts.

# Reshaping, stacking, squeezing and unsqueezing
# Often times you'll want to reshape or change the dimensions of your tensors without actually changing the values inside them.
#
# To do so, some popular methods are:
#
# Method	One-line description
# torch.reshape(input, shape)	Reshapes input to shape (if compatible), can also use torch.Tensor.reshape().
# Tensor.view(shape)	Returns a view of the original tensor in a different shape but shares the same data as the original tensor.
# torch.stack(tensors, dim=0)	Concatenates a sequence of tensors along a new dimension (dim), all tensors must be same size.
# torch.squeeze(input)	Squeezes input to remove all the dimenions with value 1.
# torch.unsqueeze(input, dim)	Returns input with a dimension value of 1 added at dim.
# torch.permute(input, dims)	Returns a view of the original input with its dimensions permuted (rearranged) to dims.





