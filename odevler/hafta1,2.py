from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt

kedi=np.array(Image.open("puf.jpg").resize((224,224))) #kedi.jpg dosyasını 224x224 boyutuna getir

kediTorch=torch.from_numpy(kedi) #numpy dizisini torch tensorüne çevir

print(kediTorch.size()) #boyutunu yazdır
print("---------------------------------------------------------")
print(kedi) #boyutunu yazdır


plt.imsave("kedi2.jpg",kediTorch[50:260,125:225,:]) #kedi2.jpg adında 50-260 ve 125-225 aralığını alarak kaydet