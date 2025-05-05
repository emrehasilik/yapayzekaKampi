import numpy as np
import torch

# Başlangıç değerleri ile tensorlar oluşturulur
x = torch.tensor(3.)
w = torch.tensor([4., 1., 2.], requires_grad=True)
b = torch.tensor(5., requires_grad=True)

print("x:", x)
print("w:", w)
print("b:", b)

# Basit lineer fonksiyon örneği: y = w[0]*x + b
y = w[0] * x + b
print("y:", y)

# Geri yayılım ile gradyanları hesaplama
y.backward()

# Hesaplanan gradyanları yazdırma
print('dy/dw:', w.grad)
print("dy/db:", b.grad)

# Eğitim verileri (inputs) ve hedefler (targets)
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

# Eğitim verilerini PyTorch tensor'ına dönüştürme
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

print("inputs:", inputs)
print("targets:", targets)

# Model için rastgele ağırlık ve bias değerleri tanımlanır
w = torch.randn(3, 2, requires_grad=True)
b = torch.randn(2, requires_grad=True)

# Lineer model fonksiyonu
def model(x):
    return x @ w + b

# Ortalama Karesel Hata (Mean Squared Error - MSE) fonksiyonu
def mse(real, pred):
    diff = real - pred
    return torch.sum(diff * diff) / diff.numel()

# Başlangıç tahminlerini hesaplama
preds = model(inputs)
print("Başlangıç tahminleri:\n", preds)

# Başlangıç kaybını hesaplama
loss = mse(targets, preds)
print("Başlangıç kayıp:", loss.item())

# Eğitim döngüsü (100 iterasyon)
for i in range(100):
    # Model tahminleri hesaplanır
    preds = model(inputs)

    # Kayıp değeri hesaplanır
    loss = mse(targets, preds)

    # Geri yayılım ile gradyanlar hesaplanır
    loss.backward()

    # Manuel parametre güncelleme (gradyan inişi)
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5

        # Gradyanlar sıfırlanır (sonraki iterasyon için)
        w.grad.zero_()
        b.grad.zero_()

# Eğitim sonrası son tahminleri hesaplama
final_preds = model(inputs)
final_loss = mse(targets, final_preds)
print("Son tahminler:\n", final_preds)
print("Son kayıp:", final_loss.item())