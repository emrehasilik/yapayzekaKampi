from torch.nn import Linear
import numpy as np
import torch

# Rastgele 1 boyutlu girdi oluştur
girdi = torch.rand(1)
print(girdi)

# Lineer katman (1 giriş, 1 çıkış)
Lineer11 = Linear(in_features=1, out_features=1)

# Katmanın ağırlık ve bias değerlerini yazdır
print("Ağırlık w : ", Lineer11.weight)
print("Y Kestiği b : ", Lineer11.bias)

# Torch ile hesaplanan lineer çıktı
print("Torch ile Lineer: ")
print(Lineer11(girdi))

# Manuel hesaplama ile karşılaştırma
print("Python ile hesapladık")
print("m*x + b,    m*girdi + b,    w*girdi + b")
print(Lineer11.weight * girdi + Lineer11.bias)

# İki farklı lineer katman oluşturma
print("-----")
Lin1 = Linear(in_features=1, out_features=5, bias=True)
print("Lin1")
print(Lin1.weight)

Lin2 = Linear(in_features=5, out_features=8)
print("Lin2")
print(Lin2.weight)

# İki katmanı ardışık şekilde kullanarak çıktıyı yazdırma
print(Lin2(Lin1(girdi)))
