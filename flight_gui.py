import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.model import predict

# Load model
weights = np.load("models/weights.npy")
bias = np.load("models/bias.npy")

# Refit scaler
df = pd.read_csv("data/Flight_Delay_Dataset_with_weather.csv")
X = df.drop(columns=["delayed"]).to_numpy()
scaler = StandardScaler()
scaler.fit(X)

# Mapping (hidden from user)
airport_codes = {
    "Toronto Pearson": 0,
    "Vancouver International": 1,
    "Montreal-Trudeau": 2
}

airline_codes = {
    "Air Canada": 0,
    "WestJet": 1,
    "Porter Airlines": 2
}

weather_codes = {
    "Clear": 0,
    "Cloudy": 1,
    "Storm": 2
}

days_of_week = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

months = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

# Set up GUI window
root = tk.Tk()
root.title("Flight Delay Predictor")
root.geometry("420x620")
root.configure(bg="#f0f8ff")

header = tk.Label(root, text="✈️ Flight Delay Predictor", font=("Helvetica", 20, "bold"), bg="#f0f8ff", pady=20)
header.pack()

entries = {}

def add_field(label_text, key, is_dropdown=False, options=None):
    label = tk.Label(root, text=label_text, font=("Helvetica", 12), bg="#f0f8ff")
    label.pack(pady=(5, 0))
    if is_dropdown:
        combo = ttk.Combobox(root, values=list(options.keys()), font=("Helvetica", 11))
        combo.pack(pady=5)
        entries[key] = (combo, options)
    else:
        entry = tk.Entry(root, font=("Helvetica", 11))
        entry.pack(pady=5)
        entries[key] = entry

# Entry fields (text)
add_field("Departure Time (0–23)", "dep_time")
add_field("Distance (in km)", "distance")

# Dropdown fields
add_field("Airline", "airline_code", is_dropdown=True, options=airline_codes)
add_field("Airport", "airport_code", is_dropdown=True, options=airport_codes)
add_field("Day of Week", "day_of_week", is_dropdown=True, options=days_of_week)
add_field("Month", "month", is_dropdown=True, options=months)
add_field("Weather", "weather", is_dropdown=True, options=weather_codes)

# Output label
output_label = tk.Label(root, text="", font=("Helvetica", 13), bg="#f0f8ff", pady=20)
output_label.pack()

# Predict logic
def predict_delay():
    try:
        user_input = [
            int(entries["dep_time"].get()),
            int(entries["distance"].get()),
            airline_codes[entries["airline_code"][0].get()],
            airport_codes[entries["airport_code"][0].get()],
            days_of_week[entries["day_of_week"][0].get()],
            months[entries["month"][0].get()],
            weather_codes[entries["weather"][0].get()]
        ]

        user_input_scaled = scaler.transform([user_input])
        prob = predict(user_input_scaled, weights, bias)[0]

        result = f"🧮 Probability of delay: {prob * 100:.2f}%\n"
        result += "🚨 Delay is likely." if prob >= 0.5 else "✅ On-time is likely."

        output_label.config(text=result)
    except Exception as e:
        messagebox.showerror("Input Error", f"Something went wrong:\n{e}")

# Button
predict_button = tk.Button(root, text="Predict Delay", command=predict_delay,
                           font=("Helvetica", 12, "bold"), bg="#007acc", fg="white", padx=20, pady=10)
predict_button.pack(pady=15)

root.mainloop()
