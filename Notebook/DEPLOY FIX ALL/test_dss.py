import numpy as np
import requests
import json

# URL endpoint API
url = "http://127.0.0.1:5000/dss-predict"

# Data yang akan dikirim dalam format JSON
data = {
    "rating_avg_attraction": 5,
    "name_attraction_cat": 2,
    "price_attraction": 3000,
    "name_city": 4,
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
print(data["predicted_place"]["name_attraction"])
print(data)
