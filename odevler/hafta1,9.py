import torch.nn as nn
from sklearn.datasets import load_breast_cancer
import torch

device = torch.device("cpu")

# Hiperparametreler
input_size = 30       # Giriş özellik sayısı
hidden_size = 500     # Gizli katman boyutu
num_classes = 2       # Çıkış sınıf sayısı (binary classification)
num_epoch = 100       # Eğitim epoch sayısı
learning_rate = 1e-4  # Öğrenme oranı (biraz artırıldı)

# Veri kümesini yükleme (X ve Y ayrıştırılıyor)
girdi, cikti = load_breast_cancer(return_X_y=True)

# NumPy dizilerini PyTorch tensor'larına dönüştürme ve cihaza taşıma
train_input = torch.from_numpy(girdi).float().to(device)  
train_output = torch.from_numpy(cikti).long().to(device)  

# Çıkış tensor'ünün boyutunu düzelt
train_output = train_output.view(-1)  # (batch_size,)

# Sinir Ağı Modeli Tanımı
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # İlk tam bağlantılı katman
        self.lrelu = nn.LeakyReLU(negative_slope=0.02) # Aktivasyon fonksiyonu
        self.fc2 = nn.Linear(hidden_size, num_classes) # İkinci tam bağlantılı katman
    
    def forward(self, input):
        out_fc1 = self.fc1(input)          # İlk tam bağlantılı katman
        out_fc1relu = self.lrelu(out_fc1)  # Aktivasyon fonksiyonu uygulanıyor
        out = self.fc2(out_fc1relu)        # İkinci tam bağlantılı katmandan geçiş
        return out

# Modelin oluşturulması ve cihaza taşınması
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Kayıp fonksiyonu ve optimizasyon
lossf = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Modelin eğitilmesi
for epoch in range(num_epoch):
    outputs = model(train_input)  # Model tahmin yapıyor
    loss = lossf(outputs, train_output)  # Kayıp hesaplanıyor
    
    optimizer.zero_grad()  # Gradyanları sıfırla
    loss.backward()  # Geri yayılım
    optimizer.step()  # Optimizasyon adımı

    # Her 10 epoch'ta bir kaybı yazdır
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}')
