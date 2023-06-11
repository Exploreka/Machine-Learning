import psycopg2
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dotenv import load_dotenv
import os

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
        tourism.id_city,
        tourism.id_attraction_cat,
        tourism.desc_attraction
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
print(tourism_rating.head())

# Menutup cursor dan koneksi

cursor.execute(tourism_with_id)

# Mengambil semua baris hasil query
rows = cursor.fetchall()

# Mendapatkan nama kolom dari cursor.description
column_names = [desc[0] for desc in cursor.description]

# Membuat DataFrame dari hasil query
tourism_with_id = pd.DataFrame(rows, columns=column_names)
print(tourism_with_id.head())

# Menutup cursor dan koneksi
cursor.close()


db.close()
user_ids = tourism_rating["id_user"].unique().tolist()

# Encode user IDs
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}

# Decoding encoded user IDs
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}

# Convert placeID to a list without duplicate values
place_ids = tourism_rating["id_attraction"].unique().tolist()

# Encoding placeID
place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}

# Decoding encoded values to placeID
place_encoded_to_place = {i: x for i, x in enumerate(place_ids)}

# Mapping userID to the user dataframe
tourism_rating["id_user"] = tourism_rating["id_user"].map(user_to_user_encoded)

# Mapping placeID to the place dataframe
tourism_rating["id_attraction"] = tourism_rating["id_attraction"].map(
    place_to_place_encoded
)

# Convert placeID to a list without duplicate values
place_ids = tourism_rating["id_attraction"].unique().tolist()

# Encoding placeID
place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}

# Decoding encoded values to placeID
place_encoded_to_place = {i: x for i, x in enumerate(place_ids)}

# Mapping userID to the user dataframe
tourism_rating["id_user"] = tourism_rating["id_user"].map(user_to_user_encoded)

# Mapping placeID to the place dataframe
tourism_rating["id_attraction"] = tourism_rating["id_attraction"].map(
    place_to_place_encoded
)

# Getting the number of users
num_users = len(user_to_user_encoded)

# Getting the number of places
num_place = len(place_encoded_to_place)

# Converting the rating to float values
tourism_rating["rating"] = tourism_rating["rating"].values.astype(np.float32)

# Minimum rating value
min_rating = min(tourism_rating["rating"])

# Maximum rating value
max_rating = max(tourism_rating["rating"])

collab_filtering = tourism_rating.sample(frac=1, random_state=42)

x = collab_filtering[["id_user", "id_attraction"]].values

# Creating the variable y to represent the ratings
y = (
    collab_filtering["rating"]
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


def recommend(id_user):
    user_id = id_user
    place_df = tourism_with_id
    df = tourism_rating

    # Mengambil sample user
    place_visited_by_user = df[df.id_user == user_id]
    place_not_visited = place_df[
        ~place_df["id_attraction"].isin(place_visited_by_user.id_attraction.values)
    ]["id_attraction"]
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
        place_visited_by_user.sort_values(by="rating", ascending=False)
        .head(5)
        .id_attraction.values
    )

    place_df_rows = place_df[place_df["id_attraction"].isin(top_place_user)]
    for row in place_df_rows.itertuples():
        print(row.name_attraction)

    print("----" * 8)
    print("Top 10 place recommendation")
    print("----" * 8)

    recommended_place = place_df[place_df["id_attraction"].isin(recommended_place_ids)]
    place_name = []
    for row in recommended_place.itertuples():
        place_name.append(row.name_attraction)
    return place_name
