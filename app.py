from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from prophet import Prophet
from transformers import pipeline
import os

app = Flask(__name__, template_folder="templates")

# Load LLM pipeline
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

# Paths for the model and data
MODEL_PATH = "prophet_weather_model.pkl"
DATA_PATH = "nyc_weather_2000_2024_celsius.csv"

# Training function
def train_weather_model():
    df = pd.read_csv(DATA_PATH)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.rename(columns={'DATE': 'ds', 'TAVG': 'y'}, inplace=True)

    model = Prophet()
    model.fit(df)
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print("Model training complete and saved.")

if __name__ == '__main__':
    train_weather_model()

# Testing function
def load_weather_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Please train the model first.")
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def predict_weather(city, start_date, days):
    model = load_weather_model()
    future_dates = pd.date_range(start=start_date, periods=days, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future_df)
    predictions_df = forecast[['ds', 'yhat']]
    predictions_df.rename(columns={'ds': 'DATE', 'yhat': 'Predicted_TAVG'}, inplace=True)
    return predictions_df

def generate_weather_summary(city, date, temp):
    prompt = f"The predicted average temperature for {city} on {date} is {temp:.2f}°C. Provide a short weather summary."
    response = llm_pipeline(prompt, max_length=50)[0]['generated_text']
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    city = request.form.get('city') 
    date = request.form.get('date') 

    if not city or not date:
        return jsonify({'error': 'Please enter both city and date'}), 400

    try:
        predictions = predict_weather(city, date, 1)
        temp = predictions.iloc[0]['Predicted_TAVG']
        weather_summary = generate_weather_summary(city, date, temp)

        return jsonify({
            'city': city,
            'date': date,
            'temperature': f"{temp:.2f}°C",
            'summary': weather_summary
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
