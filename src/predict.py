import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import predict

weights = np.load("../models/weights.npy")
bias = np.load("../models/bias.npy")
df = pd.read_csv("../data/Flight_Delay_Dataset_with_weather.csv")
X = df.drop(columns=["delayed"]).to_numpy()
scaler = StandardScaler()
scaler.fit(X)

print("Please enter flight info below:")

dep_time = int(input("Departure time (0–23): "))
distance = int(input("Distance (e.g., 500–2500): "))
airline_code = int(input("Airline code (0–2): "))
airport_code = int(input("Airport code (0–2): "))
day_of_week = int(input("Day of week (0=Mon, 6=Sun): "))
month = int(input("Month (1–12): "))
weather = int(input("Weather condition (0=Clear, 1=Cloudy, 2=Storm): "))

user_input = [dep_time, distance, airline_code, airport_code, day_of_week, month, weather]
user_input_scaled = scaler.transform([user_input])
probability = predict(user_input_scaled, weights, bias)[0]
print(f"\n🛫 Probability of flight delay: {probability * 100:.2f}%")
