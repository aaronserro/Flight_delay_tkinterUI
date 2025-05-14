# ✈️ Flight Delay Predictor

A console-based application that predicts the probability of a flight being delayed based on key input features such as departure time, weather conditions, and airline. This project was built entirely from scratch without relying on external machine learning libraries, using a custom-built ML engine and logistic regression.

---

## 📌 Overview

The Flight Delay Predictor is a lightweight machine learning tool that classifies flights as either **on-time** or **delayed** based on several features provided by the user. It uses a custom logistic regression model trained using basic gradient-based optimization techniques.

The goal of this project was to:
- Understand the mathematical foundations of machine learning
- Implement a custom ML model from scratch
- Build a functional user interface to interact with the model
- Create a practical, real-world application in the aviation sector

---

## 🚀 How It Works

The application asks the user for basic flight data (departure time, airline, weather, etc.) and outputs the probability of a delay.

### User Inputs:
- `Departure time (0–23)`
- `Flight distance (e.g., 500–2500)`
- `Airline code (0–2)`
- `Airport code (0–2)`
- `Day of the week (0 = Monday, 6 = Sunday)`
- `Month (1–12)`
- `Weather condition (0 = Clear, 1 = Cloudy, 2 = Storm)`

### Example Output:
Please enter flight info below:
Departure time (0–23): 7
Distance (e.g., 500–2500): 567
Airline code (0–2): 2
Airport code (0–2): 2
Day of week (0=Mon, 6=Sun): 5
Month (1–12): 4
Weather condition (0=Clear, 1=Cloudy, 2=Storm): 2

🛫 Probability of flight delay: 21.27%