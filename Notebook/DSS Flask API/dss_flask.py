from flask import Flask, request, jsonify
import pandas as pd
import category_encoders as ce
import pickle

app = Flask(__name__)

df = pd.read_csv("tourism_with_id.csv")
df = df.drop(["Description", "Time_Minutes", "Coordinate", "Lat", "Long"], axis=1)
# Load the trained model
model = pickle.load(open("decision_tree_model.pkl", "rb"))

# Load the encoder
encoder = ce.OrdinalEncoder(cols=["Category", "City"])


@app.route("/predict", methods=["POST"])
def predict():
    # Menerima data input dalam bentuk JSON
    input_data = request.get_json()
    input_data["Category"] = int(input_data["Category"])
    input_data["City"] = int(input_data["City"])

    # Membuat DataFrame dengan input pengguna dan menambahkan indeks
    input_df = pd.DataFrame(
        {
            "Rating": input_data["Rating"],
            "Category": input_data["Category"],
            "Price": input_data["Price"],
            "City": input_data["City"],
        },
        index=[0],
    )
    # Melakukan prediksi menggunakan model yang telah diload
    hasil_prediksi = model.predict(input_df)
    data_rekomendasi = df[df["Place_Id"] == hasil_prediksi[0]]
    data_rekomendasi = data_rekomendasi.to_dict(orient="records")[0]

    print("Nama Tempat Prediksi: \n", data_rekomendasi)
    # Mengembalikan hasil prediksi dalam bentuk JSON
    output = {"predicted_place_id": data_rekomendasi}
    print(output)

    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
