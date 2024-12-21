
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and prepare data
data = pd.read_csv('phl_exoplanet_catalog_2019.csv')

# Define features
basic_features = ['P_MASS', 'P_RADIUS', 'S_AGE', 'P_TIDAL_LOCK', 'P_HABZONE_CON', 'P_ESI']
target = 'P_HABITABLE'

# Prepare datasets
X = data[basic_features].fillna(0)
y = data[target].fillna(0)

# Train basic model
basic_model = RandomForestClassifier(n_estimators=100, random_state=42)
basic_model.fit(X, y)

# Save model
joblib.dump(basic_model, 'planet_habitability_model.pkl')
