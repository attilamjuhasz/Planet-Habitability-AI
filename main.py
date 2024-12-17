# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Loading the dataset
data = pd.read_csv('phl_exoplanet_catalog_2019.csv')

# Selecting relevant features and target
features = [
    'P_MASS', 'P_RADIUS', 'S_AGE', 'S_TIDAL_LOCK',
    'P_HABZONE_CON', 'P_ESI'
]
target = 'P_HABITABLE'

# Filtering dataset to include only selected features and target
data = data[features + [target]]

# Converting categorical columns to numeric
for col in features:
    if data[col].dtype == 'object':
        data[col] = data[col].astype('category').cat.codes

# Handling missing values
data = data.fillna(data.mean())

# Separating features (X) and target (y)
X = data[features]
y = data[target]

# Spliting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Training a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Saving the trained model for future use
joblib.dump(model, 'planet_habitability_model.pkl')

# Load the saved model
loaded_model = joblib.load('planet_habitability_model.pkl')

# New data (replace with actual values)
# Format: [P_MASS, P_RADIUS, S_AGE, S_TIDAL_LOCK, P_HABZONE_CON, P_ESI]
new_data = np.array([[1.0, 1.0, 1.0, 0, 1, 0.8]])  # Replace with actual feature values

# Makes a prediction
prediction = loaded_model.predict(new_data)
print("Prediction (0 = Not Habitable, 1 = Habitable):", prediction)
