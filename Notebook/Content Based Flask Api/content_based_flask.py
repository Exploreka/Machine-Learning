from flask import Flask, jsonify, request
import pickle
from content_based import recommend_filtering

app = Flask(__name__)


@app.route("/recommendation", methods=["POST"])
def get_recommendation():
    data = request.get_json()
    nama_tempat = data["nama_tempat"]

    # Panggil fungsi recommend_filtering dari model content-based
    recommended_places = recommend_filtering(nama_tempat)

    # Ubah hasil rekomendasi menjadi format JSON
    response = {"recommended_places": recommended_places}
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
