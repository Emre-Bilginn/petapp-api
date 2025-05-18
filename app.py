import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "pet_disease_model_mobile.pt"

# 🔁 Google Drive'dan indirme fonksiyonu
def download_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    # Dosya büyükse onay token'ı gerekir
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            response = session.get(URL, params={'id': file_id, 'confirm': value}, stream=True)
            break

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# 🔽 Model indir
if not os.path.exists(MODEL_PATH):
    print("🔽 Model indiriliyor...")
    download_from_google_drive("1LCoYcppMbLBZojgIPpoCKooYWPYEmgUp", MODEL_PATH)
    print("✅ Model başarıyla indirildi.")

# 🧠 Model tanımı ve yükleme
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.jit.load(MODEL_PATH, map_location="cpu").state_dict())
model.eval()

# 🔖 Sınıf isimleri
class_names = ['Dermatitis', 'Fungal_infections', 'Healthy', 'Hypersensitivity', 'demodicosis', 'ringworm']

# 🔄 Görsel dönüştürme işlemi
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 🧪 Tahmin API'si
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Görsel yüklenemedi'}), 400

    image = Image.open(request.files['image']).convert("RGB") # type: ignore
    image_tensor = transform(image).unsqueeze(0) # type: ignore

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()] # type: ignore

    return jsonify({'class': predicted_class})

# 🚀 Flask başlat
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
