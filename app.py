from flask import Flask, request, jsonify
import torch
from torchvision import transforms, models
from PIL import Image
import io
import torch.nn as nn
import os
import requests

app = Flask(__name__)

# 🔽 Modeli Google Drive'dan indir
def download_model_from_drive():
    file_id = "1T7d8UrjCJDWKBzSo5Qm8raiU_dMqEK4P"
    destination = "pet_disease_model_mobile.pt"
    if not os.path.exists(destination):
        print("🔽 Model Google Drive'dan indiriliyor...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        with open(destination, "wb") as f:
            f.write(response.content)
        print("✅ Model başarıyla indirildi.")

# ⬇️ İndirme işlemini başlat
download_model_from_drive()

# 📋 Etiketler
class_names = ['Dermatitis', 'Fungal_infections', 'Healthy', 'Hypersensitivity', 'demodicosis', 'ringworm']

# 🧠 Modeli yükle
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.jit.load("pet_disease_model_mobile.pt", map_location="cpu").state_dict())
model.eval()

# 🔄 Görsel dönüştürme
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya gönderilmedi'}), 400

    file = request.files['file']
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except:
        return jsonify({'error': 'Görsel okunamadı'}), 400

    input_tensor = transform(image).unsqueeze(0) # type: ignore

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()] # type: ignore

    return jsonify({
        'class': predicted_class
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
