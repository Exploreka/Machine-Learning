import requests
import json

# URL endpoint of your Flask API
url = "http://localhost:5000/recommend"

# Define the input data for recommendation
input_data = {"User_Id": 1}  # Replace with the desired user ID

# Send a POST request to the API endpoint
response = requests.post(url, json=input_data)

# Check the response status code
if response.status_code == 200:
    # Display the response data
    data = response.json()
    recommended_places = data["recommended_places"]
    print("Recommended Places:")
    for place_name in recommended_places:
        print(place_name)
else:
    print("Error:", response.status_code)
