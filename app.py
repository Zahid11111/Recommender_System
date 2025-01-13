from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd  # <-- Add this line to import pandas
import os
import time
# Initialize Flask app
app = Flask(__name__)

# Path to the model and vectorizer
MODEL_PATH = 'ann_model_improved.keras'
VECTORIZER_PATH = 'vectorizer.pkl'

# Load the trained model and vectorizer
def load_model_and_vectorizer():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Please ensure it exists in the project directory.")
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Vectorizer file not found. Please ensure it exists in the project directory.")

    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)

    with open(VECTORIZER_PATH, 'rb') as file:
        vectorizer = pickle.load(file)

    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        # Log: Start processing the request
        print("Processing request...")

        # Extract data from form
        tags = request.form.get('tags', '').lower()
        category = request.form.get('category', '').lower()
        cuisine_type = request.form.get('cuisine_type', '').lower()
        amenities = request.form.get('amenities', '').lower()
        wheelchair = request.form.get('wheelchair', 'no').lower()
        rating = float(request.form.get('rating', 0))

        print(f"Inputs received: Tags={tags}, Category={category}, ...")

        # Load dataset and model
        start_load = time.time()
        data = pd.read_csv('mauritiusDataset.csv')
        print(f"Dataset loaded in {time.time() - start_load} seconds")

        # Preprocess and filter
        filtered_data = data[data['rating'] >= rating]
        if wheelchair == 'yes':
            filtered_data = filtered_data[
                filtered_data['wheelchair_accessible'].str.contains('yes', case=False, na=False)
            ]
        if filtered_data.empty:
            return jsonify({'error': 'No recommendations found.'}), 404

        # Recommendation logic
        start_rec = time.time()
        input_text = f"{tags} {category} {cuisine_type} {amenities}"
        input_vec = vectorizer.transform([input_text]).todense()
        print(f"TF-IDF vectorization completed in {time.time() - start_rec} seconds")

        numerical_data = filtered_data[['rating', 'reviews_count', 'popularity_score']].fillna(0).to_numpy()
        input_vec_repeated = np.repeat(input_vec, numerical_data.shape[0], axis=0)
        combined_input = np.hstack([input_vec_repeated, numerical_data])
        predictions = model.predict(combined_input)
        filtered_data['score'] = predictions.flatten()

        # Sort and return top recommendations
        recommendations = filtered_data.sort_values(by='score', ascending=False).head(10)
        print(f"Recommendation generation completed in {time.time() - start_rec} seconds")
        return jsonify({'recommendations': recommendations.to_dict(orient='records')})

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT from environment or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)
