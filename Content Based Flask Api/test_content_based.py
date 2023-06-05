import requests
import json

# URL endpoint API
url = "http://127.0.0.1:5000/recommendation"

# Data yang akan dikirim dalam format JSON
data = {"nama_tempat": "Dunia Fantasi"}

# Mengirim permintaan POST dengan payload JSON
response = requests.post(url, json=data)

# Mengecek status kode respons
if response.status_code == 200:
    print("Permintaan POST berhasil.")
else:
    print("Permintaan POST gagal.")

# Mencetak respons JSON
print(response.json())
