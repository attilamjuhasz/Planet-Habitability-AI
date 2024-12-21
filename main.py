
import numpy as np
from flask import Flask, render_template, request
import joblib

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Load the trained model
model = joblib.load('planet_habitability_model.pkl')

@app.route('/purpose')
def purpose():
    return render_template('purpose.html')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None
    
    if request.method == 'POST':
        if not all(field in request.form and request.form[field] 
                  for field in ['mass', 'radius', 'age', 'tidal_lock', 'habzone', 'esi']):
            return render_template('index.html', error="Please fill out all required fields")
            
        features = [
            float(request.form['mass']),
            float(request.form['radius']),
            float(request.form['age']),
            float(request.form['tidal_lock']),
            float(request.form['habzone']),
            float(request.form['esi'])
        ]
        
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1]
        
    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
