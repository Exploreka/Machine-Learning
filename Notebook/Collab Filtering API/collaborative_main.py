from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

app = Flask(__name__)

# Load and preprocess data
tourism_rating = pd.read_csv("tourism_rating.csv")
tourism_with_id = pd.read_csv("tourism_with_id.csv")
user = pd.read_csv("user.csv")

user_ids = tourism_rating["User_Id"].unique().tolist()

# Encode user IDs
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}

# Decoding encoded user IDs
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}

# Convert placeID to a list without duplicate values
place_ids = tourism_rating["Place_Id"].unique().tolist()

# Encoding placeID
place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}

# Decoding encoded values to placeID
place_encoded_to_place = {i: x for i, x in enumerate(place_ids)}

# Mapping userID to the user dataframe
tourism_rating["User_Id"] = tourism_rating["User_Id"].map(user_to_user_encoded)

# Mapping placeID to the place dataframe
tourism_rating["Place_Id"] = tourism_rating["Place_Id"].map(place_to_place_encoded)

# Convert placeID to a list without duplicate values
place_ids = tourism_rating["Place_Id"].unique().tolist()

# Encoding placeID
place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}

# Decoding encoded values to placeID
place_encoded_to_place = {i: x for i, x in enumerate(place_ids)}

# Mapping userID to the user dataframe
tourism_rating["User_Id"] = tourism_rating["User_Id"].map(user_to_user_encoded)

# Mapping placeID to the place dataframe
tourism_rating["Place_Id"] = tourism_rating["Place_Id"].map(place_to_place_encoded)

# Getting the number of users
num_users = len(user_to_user_encoded)

# Getting the number of places
num_place = len(place_encoded_to_place)

# Converting the rating to float values
tourism_rating["Place_Ratings"] = tourism_rating["Place_Ratings"].values.astype(
    np.float32
)

# Minimum rating value
min_rating = min(tourism_rating["Place_Ratings"])

# Maximum rating value
max_rating = max(tourism_rating["Place_Ratings"])

collab_filtering = tourism_rating.sample(frac=1, random_state=42)

x = collab_filtering[["User_Id", "Place_Id"]].values

# Creating the variable y to represent the ratings
y = ( 
    collab_filtering["Place_Ratings"]
    .apply(lambda x: (x - min_rating) / (max_rating - min_rating))
    .values
)


class RecommenderNet(tf.keras.Model):
    # Function Initialization
    def __init__(self, num_users, num_place, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_place = num_place
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(  # layer embedding user
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)  # layer embedding user bias
        self.place_embedding = layers.Embedding(  # layer embeddings place
            num_place,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.place_bias = layers.Embedding(num_place, 1)  # layer embedding place bias

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])  # calling embedding layer 1
        user_bias = self.user_bias(inputs[:, 0])  # calling embedding layer 2
        place_vector = self.place_embedding(inputs[:, 1])  # calling embedding layer 3
        place_bias = self.place_bias(inputs[:, 1])  # calling embedding layer 4

        dot_user_place = tf.tensordot(user_vector, place_vector, 2)

        x = dot_user_place + user_bias + place_bias

        return tf.nn.sigmoid(x)  # sigmoid activation


# Model Initialization
model = RecommenderNet(num_users, num_place, 50)
# Call the model once to create its variables
dummy_input = tf.zeros((1, 10))  # Replace input_shape with appropriate shape
model(dummy_input)
# Load the saved model
model.load_weights("model.h5")


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    user_id = data["User_Id"]
    # place_visited_by_user = tourism_rating[tourism_rating["User_Id"] == user_id]
    # user_encoder = user_to_user_encoded.get(user_id)
    # place_not_visited = tourism_with_id[
    #     ~tourism_with_id["Place_Id"].isin(place_visited_by_user.Place_Id.values)
    # ]["Place_Id"]
    # place_not_visited = list(
    #     set(place_not_visited).intersection(set(place_to_place_encoded.keys()))
    # )
    # place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
    # user_place_array = np.hstack(
    #     (np.repeat(user_encoder, len(place_not_visited)), place_not_visited)
    # )

    # ratings = model.predict(user_place_array).flatten()
    # top_ratings_indices = ratings.argsort()[-10:][::-1]
    # recommended_place_ids = [
    #     place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices
    # ]
    place_df = tourism_with_id
    df = tourism_rating

    # Mengambil sample user
    place_visited_by_user = df[df.User_Id == user_id]
    place_not_visited = place_df[
        ~place_df["Place_Id"].isin(place_visited_by_user.Place_Id.values)
    ]["Place_Id"]
    place_not_visited = list(
        set(place_not_visited).intersection(set(place_to_place_encoded.keys()))
    )

    place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
    user_encoder = user_to_user_encoded.get(user_id)
    user_place_array = np.hstack(
        ([[user_encoder]] * len(place_not_visited), place_not_visited)
    )
    ratings = model.predict(user_place_array).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_place_ids = [
        place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices
    ]
    print("Showing recommendations for users: {}".format(user_id))
    print("===" * 9)
    print("place with high ratings from user")
    print("----" * 8)

    top_place_user = (
        place_visited_by_user.sort_values(by="Place_Ratings", ascending=False)
        .head(5)
        .Place_Id.values
    )

    place_df_rows = place_df[place_df["Place_Id"].isin(top_place_user)]
    for row in place_df_rows.itertuples():
        print(row.Place_Name)

    print("----" * 8)
    print("Top 10 place recommendation")
    print("----" * 8)

    recommended_place = place_df[place_df["Place_Id"].isin(recommended_place_ids)]
    place_name = []
    for row in recommended_place.itertuples():
        place_name.append(row.Place_Name)

    return jsonify({"recommended_places": place_name})


if __name__ == "__main__":
    app.run()
