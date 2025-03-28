import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('ad_click_dataset.csv')

# Handling missing values
df['age'] = df['age'].fillna(df['age'].median())
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
df['device_type'] = df['device_type'].fillna('Unknown')
df['ad_position'] = df['ad_position'].fillna(df['ad_position'].mode()[0])
df['browsing_history'] = df['browsing_history'].fillna('No Data')
df['time_of_day'] = df['time_of_day'].fillna(df['time_of_day'].mode()[0])

# Encoding categorical features
encoders = {}
for col in ['device_type', 'ad_position', 'browsing_history', 'time_of_day', 'gender']:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder  # Store encoder for future use

# Bin Age into groups
bins = [18, 25, 35, 45, 55, 65, 100]
labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Encode AgeGroup
age_encoder = LabelEncoder()
df['AgeGroup'] = age_encoder.fit_transform(df['AgeGroup'])
encoders['AgeGroup'] = age_encoder

# Features and target
features = ['device_type', 'AgeGroup', 'ad_position', 'browsing_history', 'gender', 'time_of_day']
X = df[features]
y = df['click']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
with open('ad_click_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save label encoders
with open('label_encoders.pkl', 'wb') as encoder_file:
    pickle.dump(encoders, encoder_file)

print("Model and encoders saved successfully!")
