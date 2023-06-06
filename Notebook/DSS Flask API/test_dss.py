import numpy as np
import requests
import json

# URL endpoint API
url = "http://127.0.0.1:5000/predict"

# Data yang akan dikirim dalam format JSON
data = {
    "Rating": 5,
    "Category": 2,
    "Price": 3000,
    "City": 4,
}

# Mengirim permintaan POST dengan payload JSON
response = requests.post(url, json=data)

# Mengecek status kode respons
if response.status_code == 200:
    print("Permintaan POST berhasil.")
else:
    print("Permintaan POST gagal.")

# Mencetak respons JSON
data = response.json()
print(data["predicted_place_id"]["Place_Name"])
