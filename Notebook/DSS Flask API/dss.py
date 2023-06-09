from flask import Flask, request, jsonify
import pandas as pd
import category_encoders as ce
import pickle
import psycopg2

app = Flask(__name__)


# Membuat koneksi ke database MySQL
db = psycopg2.connect(
    host="34.128.127.141",
    port=5432,
    user="postgres",
    password="exploreka",
    database="exploreka",
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
        tourism.id_attraction_cat
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
print(df.head())


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
    ]
]
print(df.head())
# Load the trained model
model = pickle.load(open("decision_tree_model.pkl", "rb"))

# Load the encoder
encoder = ce.OrdinalEncoder(cols=["name_attraction_cat", "name_city"])


@app.route("/predict", methods=["POST"])
def predict():
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
    data_rekomendasi = df[df["id_attraction"] == hasil_prediksi[0]]
    data_rekomendasi = data_rekomendasi.to_dict(orient="records")[0]

    print("Nama Tempat Prediksi: \n", data_rekomendasi)
    # Mengembalikan hasil prediksi dalam bentuk JSON
    output = {"predicted_place_id": data_rekomendasi}
    print(output)

    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
