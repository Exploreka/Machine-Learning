import pickle
import pandas as pd

data_filtering = pd.read_csv("./data_filtering.csv")


def recommend_filtering(nama_tempat):
    # Get the pairwsie similarity scores of all place name with given place name
    similarity = pickle.load(open("content_based_model.pkl", "rb"))
    nama_tempat_index = data_filtering[
        data_filtering["Place_Name"] == nama_tempat
    ].index[0]
    distancess = similarity[nama_tempat_index]
    # Sort place based on the similarity scores
    nama_tempat_list = sorted(
        list(enumerate(distancess)), key=lambda x: x[1], reverse=True
    )[1:20]

    recommended_nama_tempats = []
    for i in nama_tempat_list:
        recommended_nama_tempats.append([data_filtering.iloc[i[0]].Place_Name] + [i[1]])
        # print(nama_tempats.iloc[i[0]].original_title)

    return recommended_nama_tempats


recommend_filtering("Pantai Watu Kodok")
