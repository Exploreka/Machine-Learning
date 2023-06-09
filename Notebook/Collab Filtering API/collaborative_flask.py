from flask import Flask, jsonify, request
from main import recommend

app = Flask(__name__)


@app.route("/collab_recommendation", methods=["POST"])
def get_recommendation():
    data = request.get_json()
    id_user = data["id_user"]

    # Panggil fungsi recommend_filtering dari model collaborative
    recommended_places = recommend(id_user)

    # Convert recommended_places to a list
    # recommended_places = list(recommended_places)

    return jsonify({"recommended_places": recommended_places})


if __name__ == "__main__":
    app.run(debug=True)
