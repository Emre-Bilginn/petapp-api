import os
import requests
import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# 🔗 Hugging Face model bağlantısı
MODEL_PATH = "pet_disease_model_mobile.pt"
MODEL_URL = "https://huggingface.co/emrebilgin/pet-disease-model/resolve/main/pet_disease_model_mobile.pt"

# 🔽 Hugging Face'den modeli indir
def download_model(url, destination):
    r = requests.get(url)
    r.raise_for_status()
    with open(destination, "wb") as f:
        f.write(r.content)

# 🔄 Dosya yoksa indir
if not os.path.exists(MODEL_PATH):
    print("🔽 Model indiriliyor...")
    download_model(MODEL_URL, MODEL_PATH)
    print("✅ Model başarıyla indirildi.")

# 🧠 TorchScript modeli yükle
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

# 🔖 Sınıf etiketleri
class_names = ['Dermatitis', 'Fungal_infections', 'Healthy', 'Hypersensitivity', 'demodicosis', 'ringworm']

# 🔄 Görsel transform işlemi
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 🧪 Tahmin endpoint'i
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

# 🏠 Ana sayfa endpoint'i
@app.route("/", methods=["GET"])
def home():
    return "✅ PetApp API çalışıyor. Görsel tahmini için POST isteği ile /predict endpoint'ini kullanın.", 200

# 🚀 Flask sunucusunu başlat
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
