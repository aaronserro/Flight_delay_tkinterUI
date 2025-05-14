# ✈️ Flight Delay Predictor (Tkinter UI)

A desktop application that predicts the probability of a flight being delayed based on user-provided information such as departure time, weather, and airline. This tool is powered by a **custom machine learning engine** (built from scratch using Python) and features a fully interactive **Tkinter-based GUI** for a seamless user experience.

---

## 📌 Overview

The Flight Delay Predictor combines a handcrafted logistic regression model with a clean and intuitive user interface. It allows users to input common flight attributes and instantly receive a predicted probability of delay. This project was created to deepen understanding of machine learning internals while delivering a functional and realistic application in the travel and aviation domain.

---

## 🖥️ Application Features

- Fully interactive **Tkinter GUI** — no console input required
- Predicts **real-time flight delay probability**
- Powered by a **custom logistic regression model**
- Clean layout with labeled fields and intuitive form structure
- Real-time probability output in a user-friendly format

---

## 🧰 Tech Stack

| Component              | Description                            |
|------------------------|----------------------------------------|
| **Language**           | Python                                 |
| **User Interface**     | Tkinter (standard GUI library)         |
| **ML Algorithm**       | Logistic Regression (custom-built)     |
| **Optimization**       | Gradient Descent (manually implemented)|
| **Training Data**      | Structured flight data (custom Data) |

---

## 🛠️ How It Works

1. User launches the Tkinter app.
2. A form appears asking for:
   - Departure time
   - Flight distance
   - Airline
   - Airport
   - Day of the week
   - Month
   - Weather condition
3. The app processes the input through a trained logistic regression model.
4. The output is a clear message:
   **"🛫 Probability of flight delay: XX.XX%"**

---

## 🔮 Future Enhancements

- Add live flight data integration via public APIs (e.g., FAA, FlightAware)
- Deploy as a web app or cross-platform desktop app
- Implement advanced optimizers (Adam, RMSProp)
- Introduce feature selection and performance
---

## 👨‍💻 Author

Built by Aaron Serro as part of a machine learning from-scratch initiative. This project serves as a complete ML pipeline — from raw input to live prediction — all without using any external ML libraries.

---

## 📜 License

Licensed under the MIT License.
