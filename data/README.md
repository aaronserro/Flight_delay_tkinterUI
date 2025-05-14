# Flight Delay Dataset — `flights_with_weather.csv`

This dataset contains synthetic flight records used to train a machine learning model that predicts whether a flight will be delayed. A new feature — weather condition — has been added to help capture environmental factors affecting delays.

---

## 🔢 File: `flights_with_weather.csv`

Each row represents a flight, with numeric features and a binary label indicating whether the flight was delayed (`1`) or on time (`0`).

### 📚 Column Descriptions

| Column Name    | Type     | Description                                                                 |
|----------------|----------|-----------------------------------------------------------------------------|
| `dep_time`     | Integer  | Hour of departure (24-hour format, range: 0–23)                             |
| `distance`     | Integer  | Distance of the flight in miles                                             |
| `airline_code` | Integer  | Encoded airline identifier (e.g., 0 = Delta, 1 = United, 2 = American)      |
| `airport_code` | Integer  | Encoded departure airport (e.g., 0 = JFK, 1 = LAX, 2 = ORD)                 |
| `day_of_week`  | Integer  | Day of week (0 = Sunday, 1 = Monday, ..., 6 = Saturday)                     |
| `month`        | Integer  | Month of the year (1 = January, ..., 12 = December)                         |
| `weather`      | Integer  | Simulated weather conditions:<br> 0 = Clear<br> 1 = Mild<br> 2 = Severe     |
| `delayed`      | Integer  | Target label: `1` if delayed, `0` if on-time                                |

---

## 💡 Usage Notes

- This dataset is synthetic and created
