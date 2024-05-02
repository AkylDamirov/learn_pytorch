
import torch
import random
#2 Create a random tensor with shape (7, 7).
tensor = torch.rand(7,7)
# print(tensor.shape)

#3 Perform a matrix multiplication on the tensor from 2 with another
# random tensor with shape (1, 7) (hint: you may have to transpose the second tensor).
tensor_A = torch.rand(1,7)
output = torch.matmul(tensor, tensor_A.T)
# print(output)

#4 Set the random seed to 0 and do 2 & 3 over again
# torch.manual_seed(0)
# tensor = torch.rand(7,7)
# tensor_A = torch.rand(1,7)
# result = torch.matmul(tensor, tensor_A.T)
# print(result, result.shape)

#5 Speaking of random seeds, we saw how to set it with torch.manual_seed()
# but is there a GPU equivalent? (hint: you'll need to look into the documentation for torch.cuda for this one)

# torch.cuda.manual_seed(1234)

#6 Create two random tensors of shape (2, 3) and send them both to the GPU (you'll need access to a GPU for this).
# Set torch.manual_seed(1234) when creating the tensors (this doesn't have to be the GPU random seed). The output should be something like:

torch.manual_seed(1234)
if torch.cuda.is_available():
    tensor_1 = torch.rand(2,3).cuda()
    tensor_2 = torch.rand(2,3).cuda()
else:
    tensor_1 = torch.rand(2, 3)
    tensor_2 = torch.rand(2, 3)
# print(f'{tensor_1},{tensor_1.device}\n {tensor_2}, {tensor_2.device}')

#7 Perform a matrix multiplication on the tensors you created in 6 (again, you may have to adjust the shapes of one of the tensors).
output2 = torch.matmul(tensor_1, tensor_2.T)
# print(output2, output2.shape)

#8 Find the maximum and minimum values of the output of 7.
# print(f'Min-{output2.min()}')
# print(f'Max-{output2.max()}')

#9 Find the maximum and minimum index values of the output of 7.
# print(f'Position min {output2.argmin()}')
# print(f'Position max {output2.argmax()}')

#10 Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with all the 1 dimensions removed to be left
# with a tensor of shape (10). Set the seed to 7 when you create it and print out the
# first tensor and it's shape as well as the second tensor and it's shape.

# torch.manual_seed(7)
# random_tensor = torch.rand(1,1,1,10)
# new_tensor = random_tensor.squeeze()
# print(random_tensor, random_tensor.shape)
# print(new_tensor, new_tensor.shape)













