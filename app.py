from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and preprocessing artifacts
model = joblib.load('network_anomaly_model.pkl')
scaler = joblib.load('scaler.pkl')
le_dict = joblib.load('label_encoders.pkl')
feature_names = joblib.load('feature_names.pkl')

@app.route('/')
def home():
    return "Network Anomaly Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure all required features are present
        missing_cols = set(feature_names) - set(input_df.columns)
        if missing_cols:
            return jsonify({"error": f"Missing features: {list(missing_cols)}"}), 400
        
        # Reorder columns to match training
        input_df = input_df[feature_names]
        
        # Encode categorical features
        for col, le in le_dict.items():
            if col in input_df.columns:
                # Handle unseen labels by mapping them to a default or the first seen label
                # In a production app, we'd handle this more robustly
                input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_prob = model.predict_proba(input_scaled)
        
        result = {
            "prediction": "anomaly" if int(prediction[0]) == 1 else "normal",
            "confidence": float(np.max(prediction_prob[0]))
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
