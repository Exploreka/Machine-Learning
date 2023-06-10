from flask import Flask, jsonify, request
import pickle
from content_based import recommend_filtering
from collab_main import recommend
import pandas as pd
import psycopg2
import category_encoders as ce
import os
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)

db = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
)


cursor = db.cursor()

# Load and preprocess data
# tourism_rating = pd.read_csv("tourism_rating.csv")
# tourism_with_id = pd.read_csv("tourism_with_id.csv")
# user = pd.read_csv("user.csv")

tourism_rating = """
    SELECT id_user,
        id_attraction,
        rating
    FROM review_attraction;
"""

tourism_with_id = """
    SELECT tourism.id_attraction, 
        category.name_attraction_cat, 
        city.name_city, 
        tourism.name_attraction,
        tourism.price_attraction,
        tourism.rating_avg_attraction,
        tourism.id_city,
        tourism.id_attraction_cat
    FROM attraction tourism
    JOIN attraction_category category ON tourism.id_attraction_cat = category.id_attraction_cat
    JOIN city city ON tourism.id_city = city.id_city;
"""
cursor.execute(tourism_rating)

# Mengambil semua baris hasil query
rows = cursor.fetchall()

# Mendapatkan nama kolom dari cursor.description
column_names = [desc[0] for desc in cursor.description]

# Membuat DataFrame dari hasil query
tourism_rating = pd.DataFrame(rows, columns=column_names)
# print(tourism_rating.head())

# Menutup cursor dan koneksi

cursor.execute(tourism_with_id)

# Mengambil semua baris hasil query
rows = cursor.fetchall()

# Mendapatkan nama kolom dari cursor.description
column_names = [desc[0] for desc in cursor.description]

# Membuat DataFrame dari hasil query
tourism_with_id = pd.DataFrame(rows, columns=column_names)
# print(tourism_with_id.head())

# Menutup cursor dan koneksi
cursor.close()


db.close()


@app.route("/content-based-recommendation", methods=["POST"])
def get_recommendation_content():
    data = request.get_json()
    nama_tempat = data["name_attraction"]

    # Panggil fungsi recommend_filtering dari model content-based
    recommended_places = recommend_filtering(nama_tempat)

    # Ubah hasil rekomendasi menjadi format JSON
    response = {"recommended_places": recommended_places}
    return jsonify(response)


@app.route("/collab-recommendation", methods=["POST"])
def get_recommendation_colalb():
    data = request.get_json()
    id_user = data["id_user"]

    # Panggil fungsi recommend_filtering dari model collaborative
    recommended_places = recommend(id_user)

    # Convert recommended_places to a list
    # recommended_places = list(recommended_places)

    return jsonify({"recommended_places": recommended_places})





@app.route("/dss-predict", methods=["POST"])
def predict_dss():
    model = pickle.load(open("decision_tree_model.pkl", "rb"))
    # Load the encoder
    encoder = ce.OrdinalEncoder(cols=["name_attraction_cat", "name_city"])
    # print(tourism_rating.info())

    # Menerima data input dalam bentuk JSON
    input_data = request.get_json()
    input_data["name_attraction_cat"] = int(input_data["name_attraction_cat"])
    input_data["name_city"] = int(input_data["name_city"])

    # Membuat DataFrame dengan input pengguna dan menambahkan indeks
    input_df = pd.DataFrame(
        {
            "rating_avg_attraction": input_data["rating_avg_attraction"],
            "name_attraction_cat": input_data["name_attraction_cat"],
            "price_attraction": input_data["price_attraction"],
            "name_city": input_data["name_city"],
        },
        index=[0],
    )
    # Melakukan prediksi menggunakan model yang telah diload
    hasil_prediksi = model.predict(input_df)
    data_rekomendasi = tourism_with_id[
        tourism_with_id["id_attraction"] == hasil_prediksi[0]
    ]
    data_rekomendasi = data_rekomendasi.to_dict(orient="records")[0]

    # print("Nama Tempat Prediksi: \n", data_rekomendasi)
    # Mengembalikan hasil prediksi dalam bentuk JSON
    output = {"predicted_place": data_rekomendasi}
    # print(output)

    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
