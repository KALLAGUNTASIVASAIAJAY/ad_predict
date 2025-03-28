import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('ad_click_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load label encoders
with open('label_encoders.pkl', 'rb') as encoder_file:
    encoders = pickle.load(encoder_file)

# Streamlit UI
st.title("Ad Click Prediction Dashboard")

# User Input Fields
device_type = st.selectbox("Select Device Type", ["Tablet", "Mobile", "Desktop"])
ad_position = st.selectbox("Select Ad Position", ["Top", "Side", "Bottom"])
browsing_history = st.selectbox("Select Browsing History", ["Entertainment", "Shopping", "Education", "Social Media"])
gender = st.radio("Select Gender", ['Male', 'Female'])
time_of_day = st.selectbox("Select Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
age = st.number_input("Enter Age", min_value=18, max_value=100, value=25)

# Preprocess Input
age_group_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
age_bins = [18, 25, 35, 45, 55, 65, 100]
age_group = pd.cut([age], bins=age_bins, labels=age_group_labels, right=False)[0]

# Encode categorical inputs manually since we removed encoders for these fields
device_mapping = {"Tablet": 0, "Mobile": 1, "Desktop": 2}
ad_position_mapping = {"Top": 0, "Side": 1, "Bottom": 2}
browsing_mapping = {"Entertainment": 0, "Shopping": 1, "Education": 2, "Social Media": 3}
time_mapping = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}

device_encoded = device_mapping[device_type]
ad_encoded = ad_position_mapping[ad_position]
browsing_encoded = browsing_mapping[browsing_history]
gender_encoded = 0 if gender == 'Male' else 1
time_encoded = time_mapping[time_of_day]

# Encode age group using the label encoder
def encode_input(column, value):
    encoder = encoders[column]
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return -1  # Handle unseen values

age_encoded = encode_input('AgeGroup', age_group)

# Predict button
if st.button("Predict Click Probability"):
    if age_encoded == -1:
        st.error("Invalid age group detected. Some values may not have been seen during training.")
    else:
        user_input = [[device_encoded, age_encoded, ad_encoded, browsing_encoded, gender_encoded, time_encoded]]
        prediction = model.predict(user_input)
        result = "Clicked on Ad" if prediction[0] == 1 else "Did not Click on Ad"
        st.success(f"Prediction: {result}")
