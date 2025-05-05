import torch

rand1=torch.rand(2,2)
rand2=torch.rand(2,2)

print(rand1,rand2)
print(rand1+rand2)
print(rand1*rand2)
print(torch.add(rand1,rand2))


print("----------------------------------------------------")


rand3=torch.rand(10000,10000)
rand4=torch.rand(10000,10000)

print(".")
rand3.matmul(rand4)

print("...çalışıyor...")


print("-----------------------------------------------------------------------------------------------")

rand3= rand3.cuda()
rand4= rand4.cuda()

print(".")
rand3.matmul(rand4)
print("...çalışıyor22...")
