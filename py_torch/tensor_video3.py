# [Знакомство с Pytorch](https://stepik.org/lesson/1116897/step/1?unit=1128404)
import torch

print(torch.tensor([2, 3]))
print(torch.tensor([[2, 3], [4, 5]]))
print(torch.tensor([[2, 3], [4, 5]], dtype=torch.int32))
print(torch.tensor([[2, 3], [4, 5]], dtype=torch.float32))
print(torch.tensor([[2, 3], [4, 5]], dtype=torch.float32, requires_grad=True))
# print(torch.tensor([[2,3], [4,5]], device=torch.device('cuda:0')))

tensor = torch.tensor([[[2, 3], [4, 5]], [[5, 7], [8, 9]]],
                      dtype=torch.float32, requires_grad=True)

print(tensor)

# must have
print(tensor.shape)
print(tensor.size())
print(tensor.ndim)

print(tensor[0, 0, 0])  # get tensor
print(type(tensor[0, 0, 0]))  # type - tensor
print(tensor[0, 0, 0].item())  # get item
print(type(tensor[0, 0, 0].item()))  # type - float

print(torch.zeros([2, 3, 2]))
print(torch.zeros([2, 3, 2], dtype=torch.int32))

print(torch.ones([2, 3, 2]))

tensor = torch.ones([2, 3, 2])
print(torch.zeros_like(tensor))
print(torch.full_like(tensor, 7))
print(tensor.dtype)

# генерация чисел в диапазоне с шагом
print(torch.arange(2, 10, 0.5))
# диагональная матрица
print(torch.diag(torch.tensor([5, 4])))
# треугольная матрица
print(torch.eye(5))
# нижне-треугольная матрица
print(torch.tril(torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])))

# изменение формы
tensor = torch.tensor([1, 2, 3, 4])
tensor_1 = tensor.view(4, 1)
print(tensor_1)
tensor_2 = tensor.reshape(2, 2)
print(tensor_2)

# добавление осей
print(tensor)
tensor = torch.unsqueeze(tensor, 0)
print(tensor)
print(tensor.shape)

# арифметические операции
tensor = torch.tensor([1, 2, 3, 4, 5])
print(tensor * 5)
print(tensor + tensor)
# print(tensor + [1,2,3,4,5])

# математические функции
tensor = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
print(tensor.sum())
print(tensor.mean())
tensor = tensor.view([2, 3])
tensor_mean = tensor.mean(dim=1, keepdim=False)
print(tensor)
print(tensor_mean)
print(tensor_mean.shape)
tensor_mean = tensor.mean(dim=1, keepdim=True)
print(tensor_mean)
print(tensor_mean.shape)

# особенности PyTorch
# CPU и GPU
print(torch.cpu.is_available())
print(torch.cuda.is_available())

tensor = torch.tensor([1., 2., 3.], requires_grad=True)
tensor = tensor.cpu()
tensor = tensor.to('cpu')

# tensor = tensor.cuda()
# tensor = tensor.to('cuda')

# must have
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tensor.to(device)
print(tensor.device)

tensor_1 = torch.tensor([1., 2., 3.]).to('cpu')
# new_tensor = tensor + tensor_1
new_tensor = tensor + tensor_1.to(device)
print(new_tensor)
print(new_tensor.cpu())
print(new_tensor.cpu().detach())