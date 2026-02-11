from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset
data = pd.read_csv("dataset.csv")

X = data[["AQI", "Temperature", "Humidity", "Age"]]
y = data["HealthRisk"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    aqi = float(data["aqi"])
    temp = float(data["temp"])
    humidity = float(data["humidity"])
    age = float(data["age"])
    
    prediction = model.predict([[aqi, temp, humidity, age]])[0]
    
    return jsonify({"risk": prediction})

if __name__ == "__main__":
    app.run(debug=True)
