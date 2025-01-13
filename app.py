from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd  # <-- Add this line to import pandas
import os

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
        # Extract data from form
        tags = request.form.get('tags', '').lower()
        category = request.form.get('category', '').lower()
        cuisine_type = request.form.get('cuisine_type', '').lower()
        amenities = request.form.get('amenities', '').lower()
        wheelchair = request.form.get('wheelchair', 'no').lower()
        rating = float(request.form.get('rating', 0))

        # Combine input text for TF-IDF
        input_text = f"{tags} {category} {cuisine_type} {amenities}"

        # Load dataset
        data = pd.read_csv('mauritiusDataset.csv')

        # Filter data for the input rating
        filtered_data = data[data['rating'] >= rating]

        # Filter for wheelchair accessibility if required
        if wheelchair == 'yes':
            filtered_data = filtered_data[filtered_data['wheelchair_accessible'].str.contains('yes', case=False, na=False)]

        # Check if there is data to recommend
        if filtered_data.empty:
            return jsonify({'error': 'No recommendations found for the given criteria.'}), 404

        # TF-IDF transformation
        input_vec = vectorizer.transform([input_text]).todense()
        numerical_data = filtered_data[['rating', 'reviews_count', 'popularity_score']].fillna(0).to_numpy()
        input_vec_repeated = np.repeat(input_vec, numerical_data.shape[0], axis=0)
        combined_input = np.hstack([input_vec_repeated, numerical_data])

        # Make predictions
        predictions = model.predict(combined_input)
        filtered_data['score'] = predictions.flatten()

        # Get top recommendations
        recommendations = filtered_data.sort_values(by='score', ascending=False).head(10)
        display_columns = ['name', 'category', 'rating', 'address', 'imageUrls', 'latitude', 'longitude', 'url', 'popularity_score']
        recommendations = recommendations[display_columns].to_dict(orient='records')

        return jsonify({'recommendations': recommendations})

    except Exception as e:
        # Log the exception for debugging
        print(f"Error in /predict: {e}")
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT from environment or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)
