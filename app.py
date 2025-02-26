from flask import Flask, render_template, request, jsonify
import pandas as pd
from prophet import Prophet
from transformers import pipeline

app = Flask(__name__, template_folder="templates")

# Load LLM for text generation (Flan-T5)
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

# Function to predict weather
def predict_weather(city, start_date, days):
    file_path = 'nyc_weather_2000_2024_celsius.csv'  # Update with correct file path

    df = pd.read_csv(file_path)

    # Ensure correct column names and date format
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.rename(columns={'DATE': 'ds', 'TAVG': 'y'}, inplace=True)

    # Train the Prophet model
    model = Prophet()
    model.fit(df)

    # Generate future dates for prediction
    future_dates = pd.date_range(start=start_date, periods=days, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})

    forecast = model.predict(future_df)
    predictions_df = forecast[['ds', 'yhat']]
    predictions_df.rename(columns={'ds': 'DATE', 'yhat': 'Predicted_TAVG'}, inplace=True)

    return predictions_df

# Function to generate LLM-based weather summary
def generate_weather_summary(city, date, temp):
    prompt = f"The predicted average temperature for {city} on {date} is {temp:.2f}°C. Provide a short weather summary."
    response = llm_pipeline(prompt, max_length=50)[0]['generated_text']
    return response

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    city = request.form.get('city')  # Get city from form
    date = request.form.get('date')  # Get date from form

    if not city or not date:
        return jsonify({'error': 'Please enter both city and date'}), 400  # Handle missing inputs

    try:
        predictions = predict_weather(city, date, 1)
        temp = predictions.iloc[0]['Predicted_TAVG']
        
        # Get LLM-generated summary
        weather_summary = generate_weather_summary(city, date, temp)

        return jsonify({
            'city': city,
            'date': date,
            'temperature': f"{temp:.2f}°C",
            'summary': weather_summary
        })  # Send JSON response with LLM summary
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Handle errors

if __name__ == '__main__':
    app.run(debug=True)
