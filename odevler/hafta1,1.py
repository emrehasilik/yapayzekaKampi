import torch
from sklearn.datasets import fetch_california_housing
import numpy as np


# 30 elemanlı rastgele bir tensör oluştur
x = torch.rand(30)   
print(x)
print(x.size())

# FloatTensor oluşturma
temp = torch.FloatTensor([23, 24, 25, 27, 28, 29, 30.76, 31.90, 32])
print(temp)
print(temp.size())

# California Housing veri setini çek
CaliforniaHousing = fetch_california_housing()

# Veri setinin ilk 5 satırını yazdır
print(CaliforniaHousing.data)

# Veri setinin boyutunu öğrenme
print("------------------------------------------")
print(CaliforniaHousing.data.shape)  
print("--------------------------------------")
print(CaliforniaHousing.target[:2])  # İlk iki hedef değeri
print("-------------------------------------------")
print(CaliforniaHousing.data[:2])  # İlk iki satırı yazdırır



#ders 1 kodları yukarısı.............................................................................................................







