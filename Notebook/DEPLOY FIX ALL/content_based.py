import pickle
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
load_dotenv()
# data_filtering = pd.read_csv("./data_filtering.csv")
# Membuat koneksi ke database MySQL
db = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
)
# df = pd.read_csv("tourism_with_id.csv")

# Membuat objek cursor
cursor = db.cursor()

# Menjalankan perintah SQL SELECT untuk mengambil data
query = """
    SELECT tourism.id_attraction,
        category.name_attraction_cat,
        city.name_city,
        tourism.name_attraction,
        tourism.price_attraction,
        tourism.rating_avg_attraction,
        tourism.id_city,
        tourism.id_attraction_cat,
        tourism.desc_attraction
    FROM attraction tourism
    JOIN attraction_category category ON tourism.id_attraction_cat = category.id_attraction_cat
    JOIN city city ON tourism.id_city = city.id_city;
"""
cursor.execute(query)

# Mengambil semua baris hasil query
rows = cursor.fetchall()

# Mendapatkan nama kolom dari cursor.description
column_names = [desc[0] for desc in cursor.description]

# Membuat DataFrame dari hasil query
df = pd.DataFrame(rows, columns=column_names)

# Menutup cursor dan koneksi
cursor.close()
db.close()

# Melakukan operasi atau tindakan lain dengan DataFrame
# print(df.head())
df = df[
    [
        "id_attraction",
        "name_attraction",
        "price_attraction",
        "rating_avg_attraction",
        "id_city",
        "id_attraction_cat",
        "name_attraction_cat",
        "name_city",
        "desc_attraction",
    ]
]
# print(df.head())


def recommend_filtering(nama_tempat):
    # Get the pairwsie similarity scores of all place name with given place name
    similarity = pickle.load(open("content_based_model.pkl", "rb"))
    nama_tempat_index = df[df["name_attraction"] == nama_tempat].index[0]
    distancess = similarity[nama_tempat_index]
    # Sort place based on the similarity scores
    nama_tempat_list = sorted(
        list(enumerate(distancess)), key=lambda x: x[1], reverse=True
    )[1:20]

    recommended_nama_tempats = []
    for i in nama_tempat_list:
        recommended_nama_tempats.append([df.iloc[i[0]].name_attraction] + [i[1]])
        # print(nama_tempats.iloc[i[0]].original_title)

    return recommended_nama_tempats


recommend_filtering("Pantai Watu Kodok")
